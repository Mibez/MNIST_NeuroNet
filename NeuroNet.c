
//TODO: visualization
// todo: run, load, save, (quit)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <signal.h>
#include "FileReader.h"
#include "NeuroNet.h"

double Sigmoid(double in);
int InitVariables();
double ForwardPass(int p);
void BackPropagate(int p);
void Shuffle();
void InitWeights();
void PrintWeights();
double GetRandom();
void FreeMemory();
void LoadNetwork();
void SaveNetwork();
void RunNetwork();
double TrainNetwork(double learnRate, double momentum , double initWeight);
void Quit();
int StartNetwork(int inSize, int hidSize, int outSize, int tPat);
void CatchSignal(int signum);


char *pathImages = "MNIST/train-images.idx3-ubyte";
char *pathLabels = "MNIST/train-labels.idx1-ubyte";
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
int NetIsLoaded;

volatile sig_atomic_t needToStop;

void CatchSignal(int signum)
{
	if(signum == SIGINT)
	{
		needToStop = 1;
	}
}

void LoadNetwork()
{
	
	char path[50];
	printf("Please enter the path you would like to load the network from :");
	scanf("%s", path);

	printf("\nLoading network...\n");

	FILE *file;
	char *buffer;
	long fileSize = 0;

	if((file = fopen(path, "r+")) == NULL) {printf("Error opening file!\n"); return;}

	fseek(file, 0, SEEK_END);
	fileSize = ftell(file);
	rewind(file);

	buffer = (char *)malloc(sizeof(char)*fileSize);

	fread(buffer, 1, fileSize, file);

	char *tokens;
	tokens = strtok(buffer, " ");
	int eofReached = 0;
	int counter1 = 1;
	int counter2 = 0;
	weightIH[0][1] = strtod(tokens, NULL); // save the first value
	for(int j = 1; j <= hiddenSize; j++)
	{
		for(int i = 0; i <= inputSize; i++)
		{
			if((j == 1) && (i == 0)) i = 1; // we loaded first outside these loops
			tokens = strtok(NULL, " ");
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
	printf("%d input-to-hidden , %d hidde-to-output weights loaded!\n", counter1, counter2);
	if(!eofReached)
	{
		NetIsLoaded = 1;
	}
	else
	{
		printf("ERROR: file end reached before loading finished!");
	}
	fclose(file);
}

void SaveNetwork()
{
	char path[50];
	printf("Please enter a filename for your network : ");
	scanf("%s", path);
	printf("\nSaving network...\n");

	FILE *file;
	if((file = fopen(path, "w+")) == NULL) {printf("Error opening file"); return;}
	int counter1 = 0;
	int counter2 = 0;

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

void RunNetwork()
{
	Shuffle();
	int pattern = randomArray[4];
	ForwardPass(pattern);
	printf("pattern : %d \n", pattern);
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
	printf("\n correct : %f ,Network's guess : %f \n", targetOutput[pattern][1], activationO[pattern][1]);
}

void Quit()
{
	FreeMemory();
	printf("\nQuitting...\n");
}


int StartNetwork(int inSize, int hidSize, int outSize, int tPats)
{
	inputSize = inSize;
	hiddenSize = hidSize;
	outputSize = outSize;
	trainingPatterns = tPats;
	NetIsLoaded = 0;
	int errorMsg = InitVariables();


	// import training inputs and outputs
	printf("Begin file import ... ");
	errorMsg = readFile(pathImages, inputSize, 16, trainingPatterns, inputVector, 255, 0.5);
	printf("Images done, trying labels...");
	errorMsg = readFile(pathLabels, outputSize, 8, trainingPatterns, targetOutput, 10, 0.0);
	printf("Done! \n");


	// initialize the array of indices
	for(int i = 1; i < trainingPatterns; i++)
	{
		randomArray[i] = i;
	}

	return errorMsg;
}

double TrainNetwork(double learnRate, double momentum, double initWeight)
{
	needToStop = 0;

	// initialize network parameters
	eta = learnRate; // 0.2
	alpha = momentum; // 0.9
	smallWeight = initWeight; // 0.4


	error = 150.0;	// set this to 1 to enter the training loop

	// initialize the weights with random numbers if the net is not deserialized
	if(!NetIsLoaded)
	{
		InitWeights();

	}


	printf("Eta : %f alpha : %f smallWeight : %f", eta, alpha, smallWeight);
	printf("\n\nError  :\n");

		// the training loop, terminates when desired error percentage is reached
	int counter = 0;
	signal(SIGINT, CatchSignal); // exit the training loop with Ctrl + C
	while((error > 1.0) || (!needToStop))
	{
		error = 0.0;
		Shuffle();	// shuffle the array of indices

			// go through all the training samples
		for(int p = 1; p <= trainingPatterns; p++)
		{
			
			int pattern = randomArray[p];

			// go through the network and get the error
			error = ForwardPass(pattern);

			// backpropagate the error and update weights
			BackPropagate(pattern);
			if(needToStop) break;
		}
		counter++;

		// print the error every nth loop
		/*if((counter % 10) == 0)*/ printf("%f epoch : %d\n", error, counter);
	}


	printf("Error rate of %f achieved, program terminated succesfully", error);



	for(int p = 1; p < 20; p++)
	{

		double newErr = ForwardPass(p);
		printf(" Activation = %f Target = %f ", activationO[p][1], targetOutput[p][1]);

		printf(" Error : %f\n", newErr);
	}
	return error;
}

void FreeMemory()
{
	// free allocated memory

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

int InitVariables()
{
	printf("%s", "\nInitializing variables...\n");

	// memory allocation for all the weights and helper vectors, as well as the output
	inputVector = (double **) calloc(trainingPatterns+3 , sizeof(double*));
	if(!inputVector){return -1;}

	targetOutput = (double **) calloc(trainingPatterns+3 , sizeof(double*));
	if(!targetOutput) {return -1;}

	activationH = (double **) calloc(trainingPatterns+1 , sizeof(double*));
	if(!activationH) {return -1;}
	for(int i = 0; i < trainingPatterns+1; i++) //MOD
	{
		activationH[i] = (double *) calloc(hiddenSize+1 , sizeof(double));
		if(!activationH[i]) {return -1;}
	}

	activationO = (double **) calloc(trainingPatterns+1 , sizeof(double*));
	if(!activationO) {return -1;}
	for(int i = 0; i < trainingPatterns+1; i++) //MOD
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
	for(int i = 0; i < trainingPatterns+1; i++) //MOD
	{
		output[i] = (double *) calloc(outputSize+1 , sizeof(double));
		if(!output[i]) {return -1;}
	}

	randomArray = (int *) calloc(trainingPatterns+1 , sizeof(int)); // MOD
	if(!randomArray) {return -1;}

	printf("%s", "...Done\n");

	return 0;
}

double Sigmoid(double in)
{
	// the Sigmoidal activation function
	return 1.0/(1.0 + exp(-in));
}


double ForwardPass(int p)
{
	// run the inputs through the network

	// hidden layer
	for(int j = 1; j <= hiddenSize; j++)
	{
		activationH[p][j] = weightIH[0][j]; // get the neuron bias
		for(int i = 1; i <= inputSize; i++)
		{
			// sum the products of all the inputs * corresponding weights
			activationH[p][j] += inputVector[p][i] * weightIH[i][j];
		}

		activationH[p][j] = Sigmoid(activationH[p][j]); // output of each neuron of the hidden layer
	}

	// output layer
	for(int k = 1; k <= outputSize; k++)
	{
		activationO[p][k] = weightHO[k][0]; // get the neuron bias
		for(int j = 1; j <= hiddenSize; j++)
		{
			// sum the products of all the inputs * corresponding weights
			activationO[p][k] += activationH[p][j] * weightHO[k][j];
		}
		activationO[p][k] = Sigmoid(activationO[p][k]); // output of the network


		double delta = targetOutput[p][k] - activationO[p][k];
		//error += 0.5 * delta * delta;			// overall squared error
		// cross-entropy error
		error -= (targetOutput[p][k] * log(activationO[p][k]) + (1.0 - targetOutput[p][k]) * log(1.0 - activationO[p][k])); 
		//deltaO[k] = delta * activationO[p][k] * (1.0 - activationO[p][k]); // compute errors for output layer
		deltaO[k] = delta;
	}

	return error;
}

void BackPropagate(int p)
{
	// backpropagate the errors through the network and update the weights

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


void Shuffle()
{
	// shuffle the array of indices
	int newIndex = 0;
	int oldIndex = 0;
	for(int p = 1; p <= trainingPatterns; p++)
	{
		newIndex = p + GetRandom() * (trainingPatterns - p); // MODIFIED

		oldIndex = randomArray[p];
		randomArray[p] = randomArray[newIndex];
		randomArray[newIndex] = oldIndex;
	}
}

double GetRandom()
{
	// doubleify the random function
	return ((double)rand()/((double)RAND_MAX+1));
}

void InitWeights()
{

	printf("Begin weight initialization ... ");

	// fill the weights with small floating point numbers
	for(int j = 1; j <= hiddenSize; j++)
	{
		for(int i = 0; i <= inputSize; i++)
		{
			deltaWeightIH[i][j] = 0.0;
			weightIH[i][j] = 2.0 * (GetRandom() - 0.5) * smallWeight;
		}
	}
	for(int k = 1; k <= outputSize; k++)
	{
		for(int j = 0; j <= hiddenSize; j++)
		{
			deltaWeightHO[k][j] = 0.0;
			weightHO[k][j] = 2.0 * (GetRandom() - 0.5) * smallWeight;
		}
	}

	printf("%s", "...Done\n");
}

void PrintWeights()
{
	printf("%s", "\nInput to hidden weights : input vertical \n");
	for(int i = 0; i <= inputSize; i++)
	{
		for(int j = 0; j <= hiddenSize; j++)
		{
			printf("%s %f %s", " ", weightIH[j][i], " ");
		}
		printf("%s", "\n");
	}

		printf("%s", "\nHidden to Output weights : hidden vertical \n");

	for(int j = 0; j <= hiddenSize; j++)
	{
		for(int k = 0; k <= outputSize; k++)
		{
			printf("%s %f %s", " ", weightHO[k][j], " ");
		}
		printf("%s", "\n");
	}

}

