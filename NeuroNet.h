#ifndef NEURONET_H
#define NEURONET_H

int StartNetwork(int inSize, int hidSize, int outSize, int tPat);
double TrainNetwork(double learnRate, double momentum, double initWeight);
void RunNetwork();
void SaveNetwork();
void LoadNetwork();
void Quit();

#endif
