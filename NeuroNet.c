// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Title: NeuroNet.c
// Author: Miikka Lukumies
// Description: Core module and functionality for a neural network
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <signal.h>
#include "FileReader.h"
#include "NeuroNet.h"

double sigmoidFunction(double in);
int initializeVariables();
double forwardPass(int p);
void backPropagate(int p);
void shuffle();
void initializeWeights();
double getRandom();
void freeMemory();
void loadNetwork();
void saveNetwork();
void runNetwork();
double trainNetwork(double learnRate, double momentum , double initWeight);
void quitNetwork();
int startNetwork(int inSize, int hidSize, int outSize, int tPat);
void catchSignal(int signum);


char *pathImages = "data/train-ims.txt";
char *pathLabels = "data/train-labs.txt";
double eta;
double alpha;
double error;
double smallWeight;

int inputSize;
int hiddenSize;
int outputSize;
int trainingPatterns;

double **inputVector;
double **targetOutput;
double **activationH;
double **activationO;
double **weightIH;
double **weightHO;
double *sumDOW;
double *deltaH;
double *deltaO;
double **deltaWeightIH;
double **deltaWeightHO;
double **output;
int *randomArray;
int networkIsLoaded;
int networkIsTrained;

// flag for escaping training sequence
volatile sig_atomic_t needToStop;


// Custom interrupt handler, checks for SIGINT (Ctrl + C)
void catchSignal(int signum)
{
	if(signum == SIGINT)
	{
		needToStop = 1;
	}
}

// Load previously saved network weights from a user-defined file
void loadNetwork()
{
	char path[50];
	printf("Please enter the path you would like to load the network from :");
	scanf("%s", path);

	printf("\nLoading network...\n");

	FILE *file;
	char *buffer;
	long fileSize = 0;

	if((file = fopen(path, "r+")) == NULL) {printf("Error opening file!\n"); return;}

	// read the file size
	fseek(file, 0, SEEK_END);
	fileSize = ftell(file);
	rewind(file);

	// allocate memory
	buffer = (char *)malloc(sizeof(char)*fileSize);

	// read the entire content to a buffer
	fread(buffer, 1, fileSize, file);

	// begin tokenizing the weight file with whitespace as delimiter
	char *tokens;
	tokens = strtok(buffer, " ");
	int eofReached = 0;
	int counter1 = 1;
	int counter2 = 0;

	 // save the first value
	weightIH[0][1] = strtod(tokens, NULL);

	// loop through the buffer and save the rest
	for(int j = 1; j <= hiddenSize; j++)
	{
		for(int i = 0; i <= inputSize; i++)
		{
			// since we loaded first weight outside the loops
			// skip the first value
			if((j == 1) && (i == 0)) i = 1; 
			tokens = strtok(NULL, " ");

			// read until a NULL is detected -> end of file
			if(tokens != NULL)
			{
				counter1++;
				weightIH[i][j] = strtod(tokens, NULL);
			}
			else
			{
				eofReached = 1;
				break;
			}
		}
	}

	for(int k = 1; k <= outputSize; k++)
	{
		for(int j = 0;  j <= hiddenSize; j++)
		{
			tokens = strtok(NULL, " ");
			if(tokens != NULL)
			{
				counter2++;
				weightHO[k][j] = strtod(tokens,NULL);
			}
			else
			{
				eofReached = 1;
				break;
			}
		}
	}

	printf("%d input-to-hidden , %d hidden-to-output weights loaded!\n", counter1, counter2);

	// check if the read terminated prematurely
	if(!eofReached)
	{
		networkIsLoaded = 1;
		networkIsTrained = 1;
	}
	else
	{
		printf("ERROR: file end reached before loading finished!");
	}

	// clean-up
	free(buffer);
	fclose(file);
}

// Save the network weights to a user-defined filename, preferably .nn
void saveNetwork()
{
	char path[50];
	printf("Please enter a filename for your network (.nn recommended) : ");
	scanf("%s", path);
	printf("\nSaving network...\n");

	FILE *file;
	if((file = fopen(path, "w+")) == NULL) {printf("Error opening file"); return;}
	int counter1 = 0;
	int counter2 = 0;

	// loop through all the weights and save them on the file
	// with whitespace delimiter
	for(int j = 1; j <= hiddenSize; j++)
	{
		for(int i = 0; i <= inputSize; i++)
		{
			counter1++;
			fprintf(file, "%f ", weightIH[i][j]);
		}
	}

	for(int k = 1; k <= outputSize; k++)
	{
		for(int j = 0; j <= hiddenSize; j++)
		{
			counter2++;
			fprintf(file, "%f ", weightHO[k][j]);
		}
	}
	printf("%d input-to-hidden, %d hidde-to-out weights saved!\n", counter1, counter2);
	fclose(file);
}

// Runs the network with one pattern randomly selected from training set
void runNetwork()
{
	if(networkIsTrained)
	{
		// shuffle the patterns and pick one
		shuffle();
		int pattern = randomArray[4];

		// forward-propagate the selected pattern through network
		forwardPass(pattern);

		// neat little visualization
		for(int i = 0; i < inputSize; i++)
		{

			if((inputVector[pattern][i]) == -0.5)
			{
				printf(" ");
			}
			else
			{
				printf("¤");
			}

			if((i % 28) == 0) printf("\n");

		}

		// compare results
		printf("\n Correct number : %d , Network's guess : %d \n", (int)round(targetOutput[pattern][1] * 10), (int)round(activationO[pattern][1] * 10));
	}
	else
	{
		printf("\nPlease train or load the network before running!\n");
	}
}

// Clean up and quit gracefully
void quitNetwork()
{
	freeMemory();
	printf("\nQuitting...\n");
}


// Initialize the network with given parameters, import training data
int startNetwork(int inSize, int hidSize, int outSize, int tPats)
{
	networkIsTrained = 0;
	inputSize = inSize;
	hiddenSize = hidSize;
	outputSize = outSize;
	trainingPatterns = tPats;
	networkIsLoaded = 0;

	int errorMsg = initializeVariables();


	// import training inputs and outputs
	printf("Begin file import ... ");
	errorMsg = readFile(pathImages, inputSize, 16, trainingPatterns, inputVector, 255, 0.5);
	printf("Images done, labels next...");
	errorMsg = readFile(pathLabels, outputSize, 8, trainingPatterns, targetOutput, 10, 0.0);
	printf("Done! \n");


	// initialize the array of indices
	for(int i = 1; i < trainingPatterns; i++)
	{
		randomArray[i] = i;
	}

	return errorMsg;
}

// Train the network using backpropagation
double trainNetwork(double learnRate, double momentum, double initWeight)
{
	needToStop = 0;

	// initialize network parameters
	eta = learnRate;
	alpha = momentum;
	smallWeight = initWeight;


	error = 150.0;	// set this to >1 to enter the training loop

	// initialize the weights with random numbers if the net is not deserialized
	if(!networkIsLoaded && !networkIsTrained)
	{
		initializeWeights();

	}


	printf("\n\n\nLearn rate : %f momentum : %f weight init coefficient : %f", eta, alpha, smallWeight);
	printf("\nTraining the network...		Press Ctrl+C to stop\n\n\n");


	int counter = 0;
	// exit the training loop with Ctrl + C
	signal(SIGINT, catchSignal);

	// the training loop, terminates when desired error percentage is reached or on SIGINT
	while((error > 0.1) && (!needToStop))
	{
		error = 0.0;
		shuffle();	// shuffle the array of indices

			// go through all the training samples
		for(int p = 1; p <= trainingPatterns; p++)
		{

			int pattern = randomArray[p];

			// go through the network and get the error
			error = forwardPass(pattern);

			// backpropagate the error and update weights
			backPropagate(pattern);
			if(needToStop) break;
		}
		error = error / trainingPatterns; // CROSS-ENTROPY
		counter++;

		// print the cross-entropy error of the whole batch
		printf("\033[A\r\33[2K\r");
		printf("Cross-entropy error : %f Training iteration(epoch) : %d \n", error, counter);
		if(!networkIsTrained) { networkIsTrained = 1;}
	}

	if(!needToStop)
	{
		printf("Error rate of %f achieved, program terminated succesfully\n", error);
	}


	// show 20 first activations for inspecting the network performance
	if(trainingPatterns >= 20)
	{
		for(int p = 1; p < 20; p++)
		{

			forwardPass(p);
			printf(" Activation = %f Target = %f \n", activationO[p][1], targetOutput[p][1]);

		}
	}
	return error;
}

// free all dynamically allocated memory
void freeMemory()
{

	for(int i = 0; i < trainingPatterns+3; i++)
	{
		free(inputVector[i]);
		free(targetOutput[i]);
	}
	free(inputVector);
	free(targetOutput);
	free(sumDOW);
	free(deltaH);
	free(deltaO);
	free(randomArray);

	for(int i = 0; i < trainingPatterns+1; i++)
	{
		free(activationH[i]);
		free(activationO[i]);
		free(output[i]);
	}
	free(activationH);
	free(activationO);
	free(output);

	for(int i = 0; i < inputSize+1; i++)
	{
		free(weightIH[i]);
		free(deltaWeightIH[i]);
	}
	free(weightIH);
	free(deltaWeightIH);

	for(int i = 0; i < outputSize+1; i++)
	{
		free(weightHO[i]);
		free(deltaWeightHO[i]);
	}
	free(weightHO);
	free(deltaWeightHO);

}

// Allocate memory for all the network variables
int initializeVariables()
{
	printf("%s", "\nInitializing variables...");

	inputVector = (double **) calloc(trainingPatterns+3 , sizeof(double*));
	if(!inputVector){return -1;}

	targetOutput = (double **) calloc(trainingPatterns+3 , sizeof(double*));
	if(!targetOutput) {return -1;}

	activationH = (double **) calloc(trainingPatterns+1 , sizeof(double*));
	if(!activationH) {return -1;}
	for(int i = 0; i < trainingPatterns+1; i++)
	{
		activationH[i] = (double *) calloc(hiddenSize+1 , sizeof(double));
		if(!activationH[i]) {return -1;}
	}

	activationO = (double **) calloc(trainingPatterns+1 , sizeof(double*));
	if(!activationO) {return -1;}
	for(int i = 0; i < trainingPatterns+1; i++)
	{
		activationO[i] = (double*) calloc(outputSize+1 , sizeof(double));
		if(!activationO[i]) {return -1;}
	}

	weightIH = (double **) calloc(inputSize+1 , sizeof(double*));
	if(!weightIH) {return -1;}
	for(int i = 0; i < inputSize+1; i++)
	{
		weightIH[i] = (double*) calloc(hiddenSize+1 , sizeof(double));
		if(!weightIH[i]) {return -1;}
	}

	weightHO = (double **) calloc(outputSize+1 , sizeof(double*));
	if(!weightHO) {return -1;}
	for(int i = 0; i < outputSize+1; i++)
	{
		weightHO[i] = (double*) calloc(hiddenSize+1 , sizeof(double));
		if(!weightHO[i]) {return -1;}
	}

	sumDOW = (double *) calloc(hiddenSize+1 , sizeof(double));
	deltaH = (double *) calloc(hiddenSize+1 , sizeof(double));
	deltaO = (double *) calloc(outputSize+1 , sizeof(double));
	if(!sumDOW || !deltaH || !deltaO) {return -1;}

	deltaWeightIH = (double **) calloc(inputSize+1 , sizeof(double*));
	if(!deltaWeightIH) {return -1;}
	for(int i = 0; i < inputSize+1;  i++)
	{
		deltaWeightIH[i] = (double *) calloc(hiddenSize+1 , sizeof(double));
		if(!deltaWeightIH[i]) {return -1;}
	}

	deltaWeightHO = (double **) calloc(outputSize+1 , sizeof(double*));
	if(!deltaWeightHO) {return -1;}
	for(int i = 0; i < outputSize+1; i++)
	{
		deltaWeightHO[i] = (double *) calloc(hiddenSize+1 , sizeof(double));
		if(!deltaWeightHO[i]) {return -1;}
	}

	output = (double **) calloc(trainingPatterns+1 , sizeof(double*));
	if(!output) {return -1;}
	for(int i = 0; i < trainingPatterns+1; i++)
	{
		output[i] = (double *) calloc(outputSize+1 , sizeof(double));
		if(!output[i]) {return -1;}
	}

	randomArray = (int *) calloc(trainingPatterns+1 , sizeof(int));
	if(!randomArray) {return -1;}

	printf("%s", "...Done\n");

	return 0;
}

// The sigmoidal activation function
double sigmoidFunction(double in)
{
	return 1.0/(1.0 + exp(-in));
}


// Run the network for the given patterns
double forwardPass(int p)
{
	// run the inputs through the network

	// hidden layer
	for(int j = 1; j <= hiddenSize; j++)
	{
		// get the neuron bias
		activationH[p][j] = weightIH[0][j];

		for(int i = 1; i <= inputSize; i++)
		{
			// sum the products of all the inputs * corresponding weights
			activationH[p][j] += inputVector[p][i] * weightIH[i][j];
		}
		// output of each neuron of the hidden layer
		activationH[p][j] = sigmoidFunction(activationH[p][j]);
	}

	// output layer
	for(int k = 1; k <= outputSize; k++)
	{
		// get the neuron bias
		activationO[p][k] = weightHO[k][0];
		for(int j = 1; j <= hiddenSize; j++)
		{
			// sum the products of all the inputs * corresponding weights
			activationO[p][k] += activationH[p][j] * weightHO[k][j];
		}
		// output of the network
		activationO[p][k] = sigmoidFunction(activationO[p][k]);


		double delta = targetOutput[p][k] - activationO[p][k];
		//error += 0.5 * delta * delta;			// SUM SQUARED ERROR
		// CROSS-ENTROPY ERROR
		error -= (targetOutput[p][k] * log(activationO[p][k]) + (1.0 - targetOutput[p][k]) * log(1.0 - activationO[p][k])); 
		//deltaO[k] = delta * activationO[p][k] * (1.0 - activationO[p][k]); // SSE
		deltaO[k] = delta;
	}

	return error;
}

// Backpropagate the errors through the network and update the weights
void backPropagate(int p)
{
	// update hidden layer
	for(int j = 1; j <= hiddenSize; j++)
	{
		// get each neuron's contribution to the error
		sumDOW[j] = 0.0;
		for(int k = 1; k <= outputSize; k++)
		{
			sumDOW[j] += weightHO[k][j] * deltaO[k];
		}
		deltaH[j] = sumDOW[j] * activationH[p][j] * (1.0 - activationH[p][j]);

		deltaWeightIH[0][j] = eta * deltaH[j] + alpha * deltaWeightIH[0][j];
		weightIH[0][j] += deltaWeightIH[0][j];			// update neuron bias

		for(int i = 1; i <= inputSize; i++)
		{
			deltaWeightIH[i][j] = eta * inputVector[p][i] * deltaH[j] + alpha * deltaWeightIH[i][j];
			weightIH[i][j] += deltaWeightIH[i][j];		// update weights to hidden layer
		}
	}

	// update output layer
	for(int k = 1; k <= outputSize; k++)
	{
		deltaWeightHO[k][0] = eta * deltaO[k] + alpha * deltaWeightHO[k][0];
		weightHO[k][0] += deltaWeightHO[k][0];			// update bias

		for(int j = 1; j <= hiddenSize; j++)
		{
			deltaWeightHO[k][j] = eta * activationH[p][j] * deltaO[k] + alpha * deltaWeightHO[k][j];
			weightHO[k][j] += deltaWeightHO[k][j];		// update weights to output layer
		}
	}

}

// shuffle the array of indices
void shuffle()
{
	int newIndex = 0;
	int oldIndex = 0;
	for(int p = 1; p <= trainingPatterns; p++)
	{
		newIndex = p + getRandom() * (trainingPatterns - p);

		oldIndex = randomArray[p];
		randomArray[p] = randomArray[newIndex];
		randomArray[newIndex] = oldIndex;
	}
}

// The random function but double
double getRandom()
{
	return ((double)rand()/((double)RAND_MAX+1));
}

// Give initial values to all the weights
void initializeWeights()
{
	printf("Begin weight initialization ... ");

	// fill the weights with small floating point numbers
	for(int j = 1; j <= hiddenSize; j++)
	{
		for(int i = 0; i <= inputSize; i++)
		{
			deltaWeightIH[i][j] = 0.0;
			weightIH[i][j] = 2.0 * (getRandom() - 0.5) * smallWeight;
		}
	}
	for(int k = 1; k <= outputSize; k++)
	{
		for(int j = 0; j <= hiddenSize; j++)
		{
			deltaWeightHO[k][j] = 0.0;
			weightHO[k][j] = 2.0 * (getRandom() - 0.5) * smallWeight;
		}
	}

	printf("%s", "...Done\n");
}

