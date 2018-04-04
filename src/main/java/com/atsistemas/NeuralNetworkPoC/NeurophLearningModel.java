package com.atsistemas.NeuralNetworkPoC;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.NeuralNetworkType;

public class NeurophLearningModel {
	
    public static NeuralNetwork assembleNeuralNetwork() {

        Layer inputLayer = new Layer();
        inputLayer.addNeuron(new Neuron());
        inputLayer.addNeuron(new Neuron());
        inputLayer.addNeuron(new Neuron());

        Layer hiddenLayerOne = new Layer();
        hiddenLayerOne.addNeuron(new Neuron());
        hiddenLayerOne.addNeuron(new Neuron());
        hiddenLayerOne.addNeuron(new Neuron());
        hiddenLayerOne.addNeuron(new Neuron());

        Layer outputLayer = new Layer();
        outputLayer.addNeuron(new Neuron());

        NeuralNetwork ann = new NeuralNetwork();

        ann.addLayer(0, inputLayer);
        ann.addLayer(1, hiddenLayerOne);
        ConnectionFactory.fullConnect(ann.getLayerAt(0), ann.getLayerAt(1));
        ann.addLayer(2, outputLayer);
        ConnectionFactory.fullConnect(ann.getLayerAt(1), ann.getLayerAt(2));
        ConnectionFactory.fullConnect(ann.getLayerAt(0), ann.getLayerAt(ann.getLayersCount() - 1), false);

        ann.setInputNeurons(inputLayer.getNeurons());
        ann.setOutputNeurons(outputLayer.getNeurons());
        ann.setNetworkType(NeuralNetworkType.MULTI_LAYER_PERCEPTRON);
        return ann;
    }

    public static NeuralNetwork trainNeuralNetwork(NeuralNetwork ann) {
    	
        int inputSize = 3;
        int outputSize = 1;
        DataSet ds = new DataSet(inputSize, outputSize);

        DataSetRow rOne = new DataSetRow(new double[] { 00, 00, 01}, new double[] { 0 });
        ds.addRow(rOne);
        DataSetRow rTwo = new DataSetRow(new double[] { 01, 01, 01 }, new double[] { 1 });
        ds.addRow(rTwo);
        DataSetRow rThree = new DataSetRow(new double[] { 01, 00, 01 }, new double[] { 1 });
        ds.addRow(rThree);
        DataSetRow rFour = new DataSetRow(new double[] { 00, 01, 01 }, new double[] { 0 });
        ds.addRow(rFour);
        
		DataSetRow rFive = new DataSetRow(new double[] { 00, 00, 10 }, new double[] { 0 });
		ds.addRow(rFive);
		DataSetRow rSix = new DataSetRow(new double[] { 10, 10, 10 }, new double[] { 1 });
		ds.addRow(rSix);
		DataSetRow rSeven = new DataSetRow(new double[] { 10, 00, 10 }, new double[] { 1 });
		ds.addRow(rSeven);
		DataSetRow rEight = new DataSetRow(new double[] { 00, 10, 10 }, new double[] { 0 });
		ds.addRow(rEight);		
		
        BackPropagation backPropagation = new BackPropagation();
        backPropagation.setMaxIterations(1000);
        backPropagation.setMaxError(0.01);
        backPropagation.setLearningRate(0.1);

        ann.learn(ds, backPropagation);
        return ann;
    }
}