// +++++++++++++++++++++++++++++++++++++++++++++
// Title: Neuronet.h
// Author: Miikka Lukumies
// Description: Header file for NeuroNet.c
// +++++++++++++++++++++++++++++++++++++++++++++

#ifndef NEURONET_H
#define NEURONET_H

int startNetwork(int inSize, int hidSize, int outSize, int tPat);
double trainNetwork(double learnRate, double momentum, double initWeight);
void runNetwork();
void saveNetwork();
void loadNetwork();
void quitNetwork();

#endif
