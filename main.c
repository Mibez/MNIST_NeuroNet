//TODO: visualization, UI, serialization

// The definition of the neural network. Numbers of neurons in each layer
#define INPUT_SIZE 784
#define HIDDEN_SIZE 30
#define OUTPUT_SIZE 1
#define LEARN_RATE 0.2
#define MOMENTUM 0.4
#define INIT_WEIGHT 0.5


// the size of the training set
#define TRAINING_PATTERNS 60000


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include "NeuroNet.h"

void PrintHelp();
void PrintAbout();


int main()
{

	StartNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, TRAINING_PATTERNS);

	printf("Hello ! Welcome to use my Neural Network!\n\n");
	printf("Type \"help\" to get help\n");
	printf("Type \"about\" to learn more...\n");
	char msg[30];
	scanf("%s", msg);
	char help[5] = "help";
	char about[6] = "about";
	int commandAmount = 5;
	char commands[5][10] = {"load", "save", "run", "train", "exit"};
	if(strcmp(msg, help) == 0) PrintHelp();
	else if(strcmp(msg, about) == 0) PrintAbout();

	printf("Please enter a command : ");

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
					LoadNetwork();
					break;
				}
				case 1: // save
				{
					SaveNetwork();
					break;
				}
				case 2: // run
				{
					RunNetwork();
					break;
				}
				case 3: // train
				{
					TrainNetwork(LEARN_RATE, MOMENTUM, INIT_WEIGHT);
					break;
				}
				case 4: // quit
				{	
					printf("\nGoodbye\n");
					Quit();
					needToQuit = 1;
					break;
				}

			}
		}
//		printf("\033[A\r\33[2K\r"); // clear previous line
		if(!needToQuit) printf("Please enter a command : ");
	}

	return 0;
}
void PrintHelp()
{
	printf("\nHELP\n");
}

void PrintAbout()
{
	printf("\nABOUT\n");
}
