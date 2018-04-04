package com.atsistemas.NeuralNetworkPoC;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.neuroph.core.NeuralNetwork;

public class NeuralModelIntegrationTest {
	
    private NeuralNetwork ann = null;

    private void print(String input, double output, double actual) {
        System.out.println("Testing: " + input + " Expected: " + actual + " Result: " + output);
    }

    @Before
    public void annInit() {
        ann = NeurophLearningModel.trainNeuralNetwork(NeurophLearningModel.assembleNeuralNetwork());
    }

    @Test
    public void leftDisjunctTest() {
        ann.setInput(0, 0, 1);
        ann.calculate();
        print("0, 0, 1", ann.getOutput()[0], 0.0);
        //assertEquals(ann.getOutput()[0], 1.0, 0.0);
    }

    @Test
    public void rightDisjunctTest() {
        ann.setInput(1, 1, 1);
        ann.calculate();
        print("1, 1, 1", ann.getOutput()[0], 1.0);
        //assertEquals(ann.getOutput()[0], 1.0, 0.0);
    }

    @Test
    public void bothFalseConjunctTest() {
        ann.setInput(1, 0, 1);
        ann.calculate();
        print("1, 0, 1", ann.getOutput()[0], 1.0);
        //assertEquals(ann.getOutput()[0], 0.0, 0.0);
    }

    @Test
    public void bothTrueConjunctTest() {
        ann.setInput(0, 1, 1);
        ann.calculate();
        print("0, 1, 1", ann.getOutput()[0], 0.0);
        //assertEquals(ann.getOutput()[0], 0.0, 0.0);
    }
    
    @Test
    public void CustomTest() {
        ann.setInput(1, 0, 0);
        ann.calculate();
        print("1, 0, 0", ann.getOutput()[0], 1.0);
    }
    
    @Test
    public void CustomTestOne() {
        ann.setInput(0, 1, 0);
        ann.calculate();
        print("0, 1, 0", ann.getOutput()[0], 0.0);
    }
    
    @Test
    public void CustomTestTwo() {
        ann.setInput(10, 00, 10);
        ann.calculate();
        print("10, 00, 10", ann.getOutput()[0], 1.0);
    }
    
    @Test
    public void CustomTestFour() {
        ann.setInput(10, 10, 10);
        ann.calculate();
        print("10, 10, 10", ann.getOutput()[0], 1.0);
    }
    
    @Test
    public void CustomTestFive() {
        ann.setInput(00, 00, 10);
        ann.calculate();
        print("00, 10, 10", ann.getOutput()[0], 0.0);
    }
    
//    @Test
//    public void CustomTestSix() {
//        ann.setInput(0.25, 1, 1);
//        ann.calculate();
//        print("0.25, 1, 1", ann.getOutput()[0], 0.0);
//    }
//    
//    @Test
//    public void CustomTestSeven() {
//        ann.setInput(1, 0.25, 1);
//        ann.calculate();
//        print("1, 0.25, 1", ann.getOutput()[0], 1.0);
//    }
//
//    @Test
//    public void CustomTestThree() {
//        ann.setInput(0.2, 0.1);
//        ann.calculate();
//        print("0.2, 0.1", ann.getOutput()[0], 1.0);
//    }
    
    @After
    public void annClose() {
        ann = null;
    }
}