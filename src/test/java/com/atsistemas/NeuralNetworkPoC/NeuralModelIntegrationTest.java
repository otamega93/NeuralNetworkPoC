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
    public void customTestOne() {
        ann.setInput(0, 0, 1);
        ann.calculate();
        print("0, 0, 1", ann.getOutput()[0], 0.0);
        //assertEquals(ann.getOutput()[0], 1.0, 0.0);
    }

    @Test
    public void customTestTwo() {
        ann.setInput(1, 1, 1);
        ann.calculate();
        print("1, 1, 1", ann.getOutput()[0], 1.0);
        //assertEquals(ann.getOutput()[0], 1.0, 0.0);
    }

    @Test
    public void customTestThree() {
        ann.setInput(1, 0, 1);
        ann.calculate();
        print("1, 0, 1", ann.getOutput()[0], 1.0);
        //assertEquals(ann.getOutput()[0], 0.0, 0.0);
    }

    @Test
    public void customTestFour() {
        ann.setInput(0, 1, 1);
        ann.calculate();
        print("0, 1, 1", ann.getOutput()[0], 0.0);
        //assertEquals(ann.getOutput()[0], 0.0, 0.0);
    }
    
    @Test
    public void customTestFive() {
        ann.setInput(1, 0, 0);
        ann.calculate();
        print("1, 0, 0", ann.getOutput()[0], 1.0);
    }
    
    @Test
    public void customTestSix() {
        ann.setInput(0, 1, 0);
        ann.calculate();
        print("0, 1, 0", ann.getOutput()[0], 0.0);
    }
    
    @Test
    public void customTestSeven() {
        ann.setInput(10, 00, 10);
        ann.calculate();
        print("10, 00, 10", ann.getOutput()[0], 1.0);
    }
    
    @Test
    public void customTestEight() {
        ann.setInput(10, 10, 10);
        ann.calculate();
        print("10, 10, 10", ann.getOutput()[0], 1.0);
    }
    
    @Test
    public void customTestNine() {
        ann.setInput(00, 00, 10);
        ann.calculate();
        print("00, 10, 10", ann.getOutput()[0], 0.0);
    }
    
    @Test
    public void customTestTen() {
        ann.setInput(0, 0, 0);
        ann.calculate();
        print("0, 0, 0", ann.getOutput()[0], 0.0);
    }
    
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