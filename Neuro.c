//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Title: Neuro.c
// Author: Miikka Lukumies
// Description: Main interface logic for MNIST_NeuroNet
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


// The definition of the neural network. Numbers of neurons in each layer
// and learning parameters
#define INPUT_SIZE 784
#define HIDDEN_SIZE 30
#define OUTPUT_SIZE 1
#define LEARN_RATE 0.2
#define MOMENTUM 0.4
#define INIT_WEIGHT 0.5


// the size of the training set
#define TRAINING_PATTERNS 60


#include <stdio.h>
#include <string.h>
#include "NeuroNet.h"

void printHelp();
void printAbout();


int main()
{
	// allocate memory, load training sets, initialize everything
	startNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, TRAINING_PATTERNS);

	printf("Hello ! Welcome to use my Neural Network!\n\n");
	printf("Type \"help\" to get help\n");
	printf("Type \"about\" to learn more...\n");
	char msg[30];
	int commandAmount = 7;
	char commands[7][6] = {"load", "save", "run", "train", "exit", "help", "about"};
	printf("\nPlease enter a command : ");

	// get user input and check for a match to known command until quit command received
	int needToQuit = 0;
	while(!needToQuit)
	{
		int cmdIndex = 0;
		scanf("%s", msg);
		for(cmdIndex = 1; cmdIndex < commandAmount+1; cmdIndex++)
		{
			if(strcmp(msg, commands[cmdIndex-1]) == 0) break;
		}
		--cmdIndex;
		if(cmdIndex != commandAmount) // cmd found
		{

			switch(cmdIndex)
			{
				case 0: // load
				{
					loadNetwork();
					break;
				}
				case 1: // save
				{
					saveNetwork();
					break;
				}
				case 2: // run
				{
					runNetwork();
					break;
				}
				case 3: // train
				{
					trainNetwork(LEARN_RATE, MOMENTUM, INIT_WEIGHT);
					break;
				}
				case 4: // quit
				{
					printf("\nGoodbye\n");
					quitNetwork();
					needToQuit = 1;
					break;
				}
				case 5: // help
				{
					printHelp();
					break;
				}
				case 6: // about
				{
					printAbout();
					break;
				}

			}
		}
		if(!needToQuit) printf("\nPlease enter a command : ");
	}

	return 0;
}
void printHelp()
{
	printf("\n\nMNIST NeuroNet help \n");
	printf("\ntype \"about\" to learn about the commands\nand usage of this software\n\n");
}

void printAbout()
{
	printf("\n ABOUT: this is a simple implementation of\na neural network that can learn to classify handwritten\ndigits.\n\nCommands:\ntrain  - train the network until predefined limit reached\nrun    - run a single handwritten digit through the network\nload   - load the network weights from a user-defined file\nsave   - save the network weights to a user-defined file\nhelp   - get help\nabout  - display this message\nexit   - clean up everything and close the program");
}
