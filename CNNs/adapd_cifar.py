#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A Decentralized Primal-Dual Framework for Non-convex Smooth Consensus Optimization
    Paper: https://arxiv.org/abs/2107.11321
    Code provided by: Gabe Mancino-Ball

    Problem: Image Classification with CIFAR10 Dataset
"""

# Import packages
from __future__ import print_function
import argparse
import os
import sys
import time
import torch
import numpy
from mpi4py import MPI
from torchvision import datasets, transforms

# Import custom classes
from models.allcnn_c import AllCNN_C
from models.l1_regularizer import L1
from models.replace_weights import Opt

# Set up MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# Declare new method
class ADAPD:
    '''
    Proposed class for solving decentralized nonconvex consensus problems.

    :param: local_params = DICT of values that pertain to solver
    :param: mixing_matrix = Tensor of shape NxN where N=size
    :param: training_data = pytorch dataset that contains minibatchs of given size
    :param: init_weights = LIST of numpy arrays the comprise the initial wieghts of the model
    '''

    def __init__(self, local_params, mixing_matrix, training_data, init_weights):

        # Get the information about neighbor communication:
        # First, we extract the number of nodes and double check
        # this value is the same as the size of the MPI world
        # Second, we extract thr row of the mixing matrix corresponding to this agent
        # and save the weights
        self.mixing_matrix = mixing_matrix.float()
        self.num_nodes = self.mixing_matrix.shape[0]
        if self.num_nodes != size:
            sys.exit(f"Cannot match MPI size {size} with mixing matrix of shape {self.num_nodes}. ")
        self.peers = torch.where(self.mixing_matrix[rank, :] != 0)[0].tolist()
        self.peers.remove(rank)
        self.peer_weights = self.mixing_matrix[rank, self.peers].tolist()
        self.my_weight = self.mixing_matrix[rank, rank].item()

        # Here, we parse the user input from the DICT local_params
        # Model type
        if 'eta' in local_params:
            self.eta = local_params['eta']
        else:
            self.eta = 1e-2
        # Local learning rate
        if 'local_lr' in local_params:
            self.local_lr = local_params['local_lr']
        else:
            self.local_lr = 1e-2
        # Number of local steps
        if 'local_max' in local_params:
            self.local_max = int(local_params['local_max'])
        else:
            self.local_max = 1
        # One gradient?
        if 'og' in local_params:
            self.og = local_params['og']
        else:
            self.og = False
        # Local solver
        if 'local_solver' in local_params:
            self.local_solver = local_params['local_solver']
        else:
            self.local_solver = 'gd'
        # Mini-batch, should be constant across all machines
        if 'mini_batch' in local_params:
            self.mini_batch = local_params['mini_batch']
        else:
            self.mini_batch = 32
        if 'dual' in local_params:
            self.dual = local_params['dual']
        else:
            self.dual = 1 / self.eta
        # L_1 regularization coefficient
        if 'l1' in local_params:
            self.l1 = local_params['l1']
        else:
            self.l1 = 0.0
        # How often to report/save values
        if 'report' in local_params:
            self.report = local_params['report']
        else:
            self.report = 100

        # Get the CUDA device and save the data loader to be easily reference later
        self.device = torch.device(f'cuda:{rank % 8}')
        self.data_loader = training_data

        # Initialize the models
        self.model = AllCNN_C().to(self.device)

        # Initialize the updating weights rule and the training loss function
        self.replace_weights = Opt(self.model.parameters(), lr=0.1)
        self.training_loss_function = torch.nn.NLLLoss(reduction='mean')

        # Initialize the testing function to easily replace weights and not worry about overwriting values
        self.testing_model = AllCNN_C().to(self.device)

        self.testing_optimizer = Opt(self.testing_model.parameters(), lr=1e-2)

        # Initialize Local SGD
        if self.local_solver == 'gd':
            self.local_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_lr, weight_decay=0.)
        else:
            # Accelerated SGD
            self.local_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_lr, weight_decay=0.,
                                                   momentum=0.9, nesterov=True)

        # Initialize the l1 regularizer
        self.regularizer = L1(self.device)

        # Initialize all of the variables
        self.weights = [torch.tensor(init_weights[i]).to(self.device) for i in range(len(init_weights))]
        self.Y = [torch.zeros(size=p.shape).to(self.device) for p in self.model.parameters()]
        self.Z = [torch.zeros(size=p.shape).to(self.device) for p in self.model.parameters()]
        # Start global variable as the same
        self.global_variables = [torch.zeros(size=p.shape).to(self.device) for p in
                                 self.model.parameters()]
        self.neighbor_globals = [torch.zeros(size=p.shape).to(self.device) for p in
                                 self.model.parameters()]

        # Save number of parameters
        self.num_params = len(self.weights)

        # Allocate space for relevant report values: consensus, gradient,
        # iterate norm, number non-zeros, training/testing acc, compute time, etc.
        self.consensus_violation = []
        self.norm_hist = []
        self.total_optimality = []
        self.iterate_norm_hist = []
        self.nnz_at_avg = []
        self.avg_nnz = []
        self.testing_loss = []
        self.testing_accuracy = []
        self.training_loss = []
        self.training_accuracy = []
        self.compute_time = []
        self.communication_time = []
        self.total_time = []

    def solve(self, outer_iterations, training_data_full_sample, testing_data):
        '''
        Implement the method to solve the consensus problem
        '''

        # Here we use an initial barrier to make sure all of the agents start at the same time
        comm.Barrier()

        # Save the first errors using the average value - so all agents are compared fairly
        avg_weights = self.get_average_param(self.weights)
        cons, norm, total, var_norm, nnz_at_avg, avg_nnz = self.compute_optimality_criteria(avg_weights, self.weights,
                                                                       training_data_full_sample)
        self.consensus_violation.append(cons)
        self.norm_hist.append(norm)
        self.total_optimality.append(total)
        self.iterate_norm_hist.append(var_norm)
        self.nnz_at_avg.append(nnz_at_avg)
        self.avg_nnz.append(avg_nnz)

        # Time the whole algorithm with barrier at start
        t0 = time.time()
        comm.Barrier()

        ###########
        # MAIN LOOP
        ###########
        for i in range(outer_iterations):

            # TIME THIS EPOCH
            time_i = time.time()

            # SOLVE THE LOCAL PROBLEM
            if self.og:
                grads = self.get_grads(self.weights)
                self.weights = self.regularizer.forward([self.global_variables[k] - self.eta * (grads[k] + self.Y[k]) for k in
                                range(self.num_params)], self.l1)
            else:
                self.weights = self.regularizer.forward(self.subsolver(self.weights, self.global_variables, self.Y), self.l1)

            # UPDATE THE GLOBAL VARIABLES
            self.global_variables = [self.global_variables[k] - (self.eta / 2) * (
                    -self.Y[k] + self.Z[k] - (1 / self.eta) * (self.weights[k] - self.global_variables[k]) + (
                    1 / self.eta) * self.neighbor_globals[k]) for k in range(self.num_params)]

            # UPDATE DUAL VARIABLE (Y)
            self.Y = [self.Y[k] + self.dual * (self.weights[k] - self.global_variables[k]) for k in
                      range(self.num_params)]

            # STOP TIME FOR COMPUTING
            int_time1 = time.time()

            # ----- PERFORM COMMUNICATION ----- #
            comm.Barrier()
            comm_time = self.communicate_with_neighbors()
            comm.Barrier()
            # ---------------------------------- #

            # STOP TIME FOR COMPUTING
            int_time2 = time.time()

            # UPDATE DUAL VARIABLE (Z)
            self.Z = [self.Z[k] + self.dual * self.neighbor_globals[k].to(self.device) for k in
                      range(self.num_params)]
            time_i_end = time.time()

            # Save times and append information
            comp_time = round(time_i_end - int_time2 + int_time1 - time_i, 4)
            self.compute_time.append(comp_time)
            self.communication_time.append(comm_time)
            self.total_time.append(comp_time + comm_time)

            # Add a barrier so that all agents stop after each iteration
            comm.Barrier()

            # Report the relevant metrics whenever specified
            if i % self.report == 0:

                # Compute the optimality violations and append to lists
                avg_weights = self.get_average_param(self.weights)
                cons, norm, total, var_norm, nnz_at_avg, avg_nnz = self.compute_optimality_criteria(avg_weights,
                                                                                                    self.weights,
                                                                                                    training_data_full_sample)
                self.consensus_violation.append(cons)
                self.norm_hist.append(norm)
                self.total_optimality.append(total)
                self.iterate_norm_hist.append(var_norm)
                self.nnz_at_avg.append(nnz_at_avg)
                self.avg_nnz.append(avg_nnz)

                # Perform the local training and testing computations
                train_loss, train_acc = self.test(avg_weights, self.data_loader)
                self.training_loss.append(train_loss)
                self.training_accuracy.append(train_acc)
                test_loss, test_acc = self.test(avg_weights, testing_data)
                self.testing_loss.append(test_loss)
                self.testing_accuracy.append(test_acc)

                # Print training information
                if rank == 0:

                    # First iteration, print headings, then print the values
                    if i == 0:
                        print("{:<10} | {:<7} | {:<13} | {:<15} | {:<15} | {:<12} | {:<6}".format("Iteration", "Epoch",
                                                "Stationarity", "Train (L / A)", "Test (L / A)", "Avg Density", "Time"))
                    print("{:<10} | {:<7} | {:<13} | {:<15} | {:<15} | {:<12} | {:<6}".format(i,
                                                round((self.local_max * i * self.mini_batch) / (50000 // size), 2), round(total, 4),
                                                f"{round(train_loss, 4)} / {round(train_acc, 2)}",
                                                f"{round(test_loss, 4)} / {round(test_acc, 2)}",
                                                round(avg_nnz, 6),round(time.time() - t0, 1)))

        # End total training time
        t1 = time.time() - t0
        if rank == 0:
            closing_statement = f' Training finished '
            print('\n' + closing_statement.center(50, '-'))
            print(f'[TOTAL TIME] {round(t1, 2)}')

        # Return the training time
        return t1

    def get_grads(self, current_weights):
        '''Get a local gradient'''

        # Update parameters
        self.replace_weights.step(current_weights, self.device)

        # Set model to training mode
        self.model.train()

        # Choose one random sample
        for batch_idx, (data, target) in enumerate(self.data_loader):
            # Print errors
            torch.autograd.set_detect_anomaly(True)

            # Zero out gradients
            self.replace_weights.zero_grad()

            # Convert data to CUDA if possible
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass of the model
            out = self.model(data)
            loss = (1 / self.num_nodes) * self.training_loss_function(out, target)

            # Compute the gradients
            loss.backward()

            # Return sample gradient
            return [p.grad.data.detach().to(self.device) for p in self.model.parameters()]

    def subsolver(self, init_guess, current_global_variable, current_dual_variable):
        '''Solve the local problem by ADAM'''

        # Take the first step
        self.replace_weights.zero_grad()
        self.replace_weights.step(init_guess, self.device)

        # Reset momentum if necessary
        if self.local_solver == 'gd':
            self.local_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_lr, weight_decay=0.)
        else:
            # Accelerated SGD
            self.local_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_lr, weight_decay=0.,
                                                   momentum=0.9, nesterov=True)

        # Do SGD steps
        for i in range(self.local_max):

            # Loop over mini-batches
            for batch_idx, (data, target) in enumerate(self.data_loader):

                # Zero SGD gradients
                self.local_optimizer.zero_grad()

                # Convert data to CUDA if possible
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass of the model
                out = self.model(data)
                loss = (1 / self.num_nodes) * self.training_loss_function(out, target)

                # Compute the inner product
                for ind, p in enumerate(self.model.parameters()):
                    loss = loss + torch.sum(current_dual_variable[ind] * (p - current_global_variable[ind]))
                    loss = loss + (1 / (2 * self.eta)) * torch.norm(p - current_global_variable[ind], p='fro') ** 2

                # Compute the gradients
                loss.backward()

                # Take a step
                self.local_optimizer.step()

                # ONLY DO ONE STOCHASTIC GRADIENT STEP
                break

        return [p.data.detach().to(self.device) for p in self.model.parameters()]

    def communicate_with_neighbors(self):

        # TIME IT
        time0 = MPI.Wtime()

        # ----- LOOP OVER PARAMETERS ----- #
        for pa in range(self.num_params):

            # DEFINE VARIABLE TO SEND
            send_data = self.global_variables[pa].cpu().detach().numpy()
            recv_data = numpy.empty(shape=((len(self.peers),) + self.global_variables[pa].shape), dtype=numpy.float32)

            # SET UP REQUESTS TO INSURE CORRECT SENDS/RECVS
            recv_request = [MPI.REQUEST_NULL for ind in range(int(2 * len(self.peers)))]

            # SEND THE DATA
            for ind, peer_id in enumerate(self.peers):
                # Send the data
                recv_request[ind + len(self.peers)] = comm.Isend(send_data, dest=peer_id)

            # RECEIVE THE DATA
            for ind, peer_id in enumerate(self.peers):
                # Receive the data
                recv_request[ind] = comm.Irecv(recv_data[ind, :], source=peer_id)

            # HOLD UNTIL ALL COMMUNICATIONS COMPLETE
            MPI.Request.waitall(recv_request)

            # SCALE CURRENT WEIGHTS
            self.neighbor_globals[pa] = (1 - self.my_weight) * self.global_variables[pa]

            # Update global variables
            for ind in range(len(self.peers)):
                self.neighbor_globals[pa] -= (self.peer_weights[ind] * torch.tensor(recv_data[ind, :]).to(self.device))

        return round(MPI.Wtime() - time0, 4)

    def get_average_param(self, list_of_params):
        '''Perform ALLREDUCE of neighbor parameters'''

        # Save information to blank list
        output_list_of_parameters = [None] * len(list_of_params)

        # Loop over the parameters
        for pa in range(self.num_params):

            # Prep send and receive to be numpy arrays
            send_data = list_of_params[pa].cpu().detach().numpy()
            recv_data = numpy.empty(shape=(list_of_params[pa].shape), dtype=numpy.float32)

            # Barriers and note that the allreduce operations is summation!
            comm.Barrier()
            comm.Allreduce(send_data, recv_data)
            comm.Barrier()

            # Save information by dividing by number of agents and converting to tensor
            output_list_of_parameters[pa] = (1 / self.num_nodes) * torch.tensor(recv_data).to(self.device)

        return output_list_of_parameters

    def compute_optimality_criteria(self, avg_weights, local_weights, training_data_full_sample):
        '''
        Compute the relevant metrics for this problem

        :param avg_weights: LIST of average weights
        :param local_weights: LIST of local weights
        :param training_data_full_sample: data loader with full gradient size
        :return:
        '''

        # Compute consensus for this agent
        local_violation = sum([numpy.linalg.norm(
            local_weights[i].cpu().numpy().flatten() - avg_weights[i].cpu().numpy().flatten(), ord=2) ** 2 for i in
                               range(len(local_weights))])

        # Compute the norm of the iterate to save in case consensus is large
        avg_weight_norm = sum([numpy.linalg.norm(avg_weights[i].cpu().numpy().flatten(), ord=2) ** 2 for i in
                               range(len(avg_weights))])

        # Compute the gradient at the average solution on this dataset:
        # 1. Replace the model params
        # 2. Forward pass, backward pass to have gradient
        # 3. Compute the stationarity violation
        self.replace_weights.step(avg_weights, self.device)
        self.model.train()
        grads = [torch.zeros(size=p.shape).to(self.device) for p in self.model.parameters()]
        for batch_idx, (data, target) in enumerate(training_data_full_sample):

            # Print errors (just in case) and zero out the gradient
            torch.autograd.set_detect_anomaly(True)
            self.replace_weights.zero_grad()
            data, target = data.to(self.device), target.to(self.device)

            # Forward and backward pass of the model; scale by (1 / N) to line up with average
            out = self.model(data)
            loss = (1 / self.num_nodes) * self.training_loss_function(out, target)
            loss.backward()

            # Save gradients
            grads = [grads[ind] + p.grad.data.detach().to(self.device) for ind, p in
                     enumerate(self.model.parameters())]

        # Get the average gradient by doing all_reduce and then compute the stationarity violation at the average point
        avg_grads = self.get_average_param(grads)
        stationarity1 = self.regularizer.forward([avg_weights[pa] - avg_grads[pa] for pa in range(self.num_params)], self.l1)
        stationarity = numpy.concatenate([avg_weights[pa].detach().cpu().numpy().flatten()
                            - stationarity1[pa].detach().cpu().numpy().flatten() for pa in range(self.num_params)])
        global_norm = numpy.linalg.norm(stationarity, ord=2) ** 2

        # Before sending, also get then number of non-zeros for this agent and this average
        _, local_nnz_ratio = self.regularizer.number_non_zeros(local_weights)
        _, nnz_at_average = self.regularizer.number_non_zeros(avg_weights)

        # Perform all-reduce to have sum of local violations, i.e. Frobenius norm of consensus
        array_to_send = numpy.array([local_violation, local_nnz_ratio])
        recv_array = numpy.empty(shape=array_to_send.shape)
        comm.Barrier()
        comm.Allreduce(array_to_send, recv_array)
        comm.Barrier()

        # return consensus, gradient, total optimality, iterate history,
        # local number non-zeros, number nonzeros at everate, and average number of nonzeros
        return recv_array[0], global_norm, recv_array[0] + global_norm, avg_weight_norm, \
               nnz_at_average, (1 / size) * recv_array[1]

    def test(self, weights, testing_data):
        '''Test the data using the average weights'''

        self.testing_optimizer.zero_grad()
        self.testing_optimizer.step(weights, self.device)
        self.testing_model.eval()

        # Create separate testing loss for testing data
        loss_function = torch.nn.NLLLoss(reduction='sum')

        # Allocate space for testing loss and accuracy
        test_loss = 0
        correct = 0

        # Do not compute gradient with respect to the testing data
        with torch.no_grad():

            # Loop over testing data
            for data, target in testing_data:

                # Use CUDA
                data, target = data.to(self.device), target.to(self.device)

                # Evaluate the model on the testing data
                output = self.testing_model(data)
                test_loss += loss_function(output, target).item()

                # Gather predictions on testing data
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Compute number of testing data points
        num_test_points = int(len(testing_data.dataset) / size)

        # PERFORM ALL REDUCE TO HAVE AVERAGE
        array_to_send = numpy.array([correct, num_test_points, test_loss])
        recv_array = numpy.empty(shape=array_to_send.shape)

        # Barrier
        comm.Barrier()
        comm.Allreduce(array_to_send, recv_array)
        comm.Barrier()

        # Save loss and accuracy
        test_loss = recv_array[2] / recv_array[1]
        testing_accuracy = 100 * recv_array[0] / recv_array[1]

        return test_loss, testing_accuracy


if __name__=='__main__':

    # Parse user input
    parser = argparse.ArgumentParser(description='Testing ADAPD on CIFAR-10 dataset.')

    parser.add_argument('--epochs', type=int, default=1000, help='Total number of communication rounds.')
    parser.add_argument('--og', type=bool, default=False, help='One or multiple gradients?')
    parser.add_argument('--lr', type=float, default=1e-2, help='Local learning rate.')
    parser.add_argument('--l1', type=float, default=0.0, help='L-1 Regularizer.')
    parser.add_argument('--comm_pattern', type=str, default='ring', choices=['ring', 'random', 'complete'], help='Communication pattern.')
    parser.add_argument('--init', type=str, default='normal', choices=['he', 'normal'], help='Initialization type.')
    parser.add_argument('--trial', type=int, default=1, help='Which starting variables to use.')
    parser.add_argument('--eta', type=float, default=1.0, help='Penalty parameter')
    parser.add_argument('--local_solver', type=str, default='gd', choices=('gd', 'agd'), help='Local subsolver.')
    parser.add_argument('--mini_batch', type=int, default=64, help='Mini-batch size.')
    parser.add_argument('--dual', type=float, default=1e-2, help='Dual step-size.')
    parser.add_argument('--local_max', type=int, default=1, help='Integer number of local steps.')
    parser.add_argument('--report', type=int, default=100, help='How often to report criteria.')

    # Create callable argument
    args = parser.parse_args()

    # Create transform for data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load the data
    num_samples = 50000 // size
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=False,
                         transform=transform),
        batch_size=args.mini_batch, sampler=torch.utils.data.SubsetRandomSampler(
            [i for i in range(int(rank * num_samples), int((rank + 1) * num_samples))]))

    # Load data to be used to compute full gradient with neighbors
    optimality_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=False,
                         transform=transform),
        batch_size=num_samples, sampler=torch.utils.data.SubsetRandomSampler(
            [i for i in
             range(int(rank * num_samples), int((rank + 1) * num_samples))]))  # Difference is in number of samples!!

    # Load the testing data
    num_test = 10000 // size
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, transform=transform),
        batch_size=num_test, sampler=torch.utils.data.SubsetRandomSampler(
            [i for i in range(int(rank * num_test), int((rank + 1) * num_test))]))

    # Load communication matrix and initial weights
    mixing_matrix = numpy.load(f'mixing_matrices/{args.comm_pattern}_{size}.dat', allow_pickle=True)
    mixing_matrix = torch.tensor(mixing_matrix)
    init_weights = [numpy.load(os.path.join(os.getcwd(), f'init_weights_{args.init}/trial{args.trial}/rank{rank}/layer{l}.dat'), allow_pickle=True)
                    for l in range(28)]

    # Print training information
    if rank == 0:
        if args.og:
            opening_statement = f' ADAPD-OG on CIFAR-10 '
        else:
            opening_statement = f' ADAPD on CIFAR-10 '
        print('\n' + opening_statement.center(50, '-'))
        print(f'[GRAPH INFO] {size} agents | eigenvalues = {torch.sort(torch.eig(mixing_matrix)[0][:, 0])[0]}')
        print(f'[TRAINING INFO] mini-batch = {args.mini_batch} | learning rate = {args.eta}\n')

    # Barrier before training
    comm.Barrier()

    # Declare and train!
    algo_params = {'local_lr': args.lr, 'mini_batch': args.mini_batch, 'eta': args.eta,
                   'local_max': args.local_max, 'report': args.report, 'local_solver': args.local_solver,
                   'l1': args.l1, 'og': args.og, 'dual': args.dual}
    solver = ADAPD(algo_params, mixing_matrix, train_loader, init_weights)
    algo_time = solver.solve(args.epochs, optimality_loader, test_loader)

    # Save information
    if args.og:
        method = 'adapd_og'
    else:
        method = 'adapd'

    # Make directory for both the dataset and the method and the model
    try:
        os.mkdir(os.path.join(os.getcwd(), f'results/'))
    except:
        # Main storage already exists already exists
        pass
    try:
        os.mkdir(os.path.join(os.getcwd(), f'results/{method}'))
    except:
        # Method already exists
        pass
    try:
        os.mkdir(os.path.join(os.getcwd(), f'results/{method}/trial{args.trial}'))
    except:
        # Trial already exists
        pass
    try:
        os.mkdir(os.path.join(os.getcwd(), f'results/{method}/trial{args.trial}/{args.comm_pattern}{size}'))
    except:
        # Graph and size already exists
        pass
    try:
        os.mkdir(os.path.join(os.getcwd(), f'results/{method}/trial{args.trial}/{args.comm_pattern}{size}/{args.mini_batch}'))
    except:
        # Mini-batch already exists
        pass

    # Save path
    path = os.path.join(os.getcwd(), f'results/{method}/trial{args.trial}/{args.comm_pattern}{size}/{args.mini_batch}')

    # Save information via numpy
    if rank == 0:
        numpy.savetxt(
            f'{path}/test_loss_eta{args.eta}_solver{args.local_solver}_max{args.local_max}_stepsize{args.lr}_dual{args.dual}_l1{args.l1}.txt',
            solver.testing_loss, fmt='%.7f')
        numpy.savetxt(
            f'{path}/test_acc_eta{args.eta}_solver{args.local_solver}_max{args.local_max}_stepsize{args.lr}_dual{args.dual}_l1{args.l1}.txt',
            solver.testing_accuracy, fmt='%.7f')
        numpy.savetxt(
            f'{path}/train_loss_eta{args.eta}_solver{args.local_solver}_max{args.local_max}_stepsize{args.lr}_dual{args.dual}_l1{args.l1}.txt',
            solver.training_loss, fmt='%.7f')
        numpy.savetxt(
            f'{path}/train_acc_eta{args.eta}_solver{args.local_solver}_max{args.local_max}_stepsize{args.lr}_dual{args.dual}_l1{args.l1}.txt',
            solver.training_accuracy, fmt='%.7f')
        numpy.savetxt(
            f'{path}/total_opt_eta{args.eta}_solver{args.local_solver}_max{args.local_max}_stepsize{args.lr}_dual{args.dual}_l1{args.l1}.txt',
            solver.total_optimality, fmt='%.7f')
        numpy.savetxt(
            f'{path}/consensus_eta{args.eta}_solver{args.local_solver}_max{args.local_max}_stepsize{args.lr}_dual{args.dual}_l1{args.l1}.txt',
            solver.consensus_violation, fmt='%.7f')
        numpy.savetxt(
            f'{path}/norm_hist_eta{args.eta}_solver{args.local_solver}_max{args.local_max}_stepsize{args.lr}_dual{args.dual}_l1{args.l1}.txt',
            solver.norm_hist, fmt='%.7f')
        numpy.savetxt(
            f'{path}/iterate_hist_eta{args.eta}_solver{args.local_solver}_max{args.local_max}_stepsize{args.lr}_dual{args.dual}_l1{args.l1}.txt',
            solver.iterate_norm_hist, fmt='%.7f')
        numpy.savetxt(
            f'{path}/total_time_eta{args.eta}_solver{args.local_solver}_max{args.local_max}_stepsize{args.lr}_dual{args.dual}_l1{args.l1}.txt',
            solver.total_time, fmt='%.7f')
        numpy.savetxt(
            f'{path}/comm_time_eta{args.eta}_solver{args.local_solver}_max{args.local_max}_stepsize{args.lr}_dual{args.dual}_l1{args.l1}.txt',
            solver.communication_time, fmt='%.7f')
        numpy.savetxt(
            f'{path}/comp_time_eta{args.eta}_solver{args.local_solver}_max{args.local_max}_stepsize{args.lr}_dual{args.dual}_l1{args.l1}.txt',
            solver.compute_time, fmt='%.7f')

    # Barrier at end so all agents stop this script before moving on
    comm.Barrier()