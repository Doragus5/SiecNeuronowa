import java.util.Random;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.io.File;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;


//import java.io.*;
public class Network {
    private Matrix[] activations;
    private Matrix[] weightedSums;
    private Matrix[] biases;
    private Matrix[] weights;
    private Matrix[] activationDerivatives;
    private Matrix[] weightedSumDerivatives;
    private Matrix[] biasDerivatives;
    private Matrix[] weightDerivatives;
    private Matrix[] activationCummulativeDerivatives;
    private Matrix[] weightedSumCummulativeDerivatives;
    private Matrix[] biasCummulativeDerivatives;
    private Matrix[] weightCummulativeDerivatives;

    public final int inputSize;
    public final int innerLayers;
    private int numberOfTests = 0;

    private static final DoubleUnaryOperator sigmoid = (double x) -> {
        return 1.0 / (1.0 + Math.exp(-x));
    };

    private static final DoubleUnaryOperator sigmoidDerivative = (double x) -> {
        return sigmoid.applyAsDouble(x) * (1.0 - sigmoid.applyAsDouble(x));
    };

    private static final DoubleUnaryOperator initializeBias = (double x) -> {
        return Math.random() * 20.0 - 10.0;
    };

    private static final DoubleBinaryOperator initializeWeights = (double x, double y) -> {
        Random random = new Random();
        return random.nextGaussian() * Math.sqrt(2.0 / (x + y));
    };
    

    Network(int[] innerLayers, int inputLayer, int outputLayer) {
        inputSize = inputLayer;
        this.innerLayers = innerLayers.length;
        weightedSums =                      new Matrix[innerLayers.length + 2];
        activations =                       new Matrix[innerLayers.length + 2];
        biases =                            new Matrix[innerLayers.length + 2];
        weightedSumCummulativeDerivatives = new Matrix[innerLayers.length + 2];
        activationCummulativeDerivatives =  new Matrix[innerLayers.length + 2];
        biasCummulativeDerivatives =        new Matrix[innerLayers.length + 2];
        weightedSumDerivatives =            new Matrix[innerLayers.length + 2];
        activationDerivatives =             new Matrix[innerLayers.length + 2];
        biasDerivatives =                   new Matrix[innerLayers.length + 2];
        weights =                           new Matrix[innerLayers.length + 1];
        weightCummulativeDerivatives =      new Matrix[innerLayers.length + 1];
        weightDerivatives =                 new Matrix[innerLayers.length + 1];
        initializeMatrixes(innerLayers, inputLayer, outputLayer);
    }
    
    private void initializeMatrixes(int[] innerLayers, int inputLayer, int outputLayer) {
        activations[0] = new Matrix(inputLayer, 1);
        activations[innerLayers.length + 1] = new Matrix(outputLayer);
        biases[0] = new Matrix(inputLayer);
        biases[innerLayers.length + 1] = new Matrix(outputLayer);
        weights[0] = new Matrix(inputLayer, innerLayers[0]);
        weights[innerLayers.length] = new Matrix(innerLayers[innerLayers.length - 1], outputLayer);
        weights[innerLayers.length].applyFunction(initializeWeights);

        for (int i = 1; i < innerLayers.length + 1; i++) {
            activations[i] = new Matrix(innerLayers[i - 1]);
            biases[i] = new Matrix(innerLayers[i - 1]);
            biases[i].applyFunction(initializeBias);
            weightedSums[i] = new Matrix(innerLayers[i - 1]);
        }
        for (int i = 1; i < innerLayers.length; i++) {
            weights[i] = new Matrix(innerLayers[i - 1], innerLayers[i]);
            weights[i].applyFunction(initializeWeights);
        }
        initializeDerivatives();
        initializeCummulativeDerivatives();
    }
    
    private void initializeDerivatives() {
        for (int i = 1; i < innerLayers + 1; i++) {
            activationDerivatives[i] = new Matrix(activations[i].rows);
            weightedSumDerivatives[i] = new Matrix(weightedSums[i].rows);
            biasDerivatives[i] = new Matrix(biases[i].rows);
        }
        for (int i = 1; i < innerLayers; i++) {
            weightDerivatives[i] = new Matrix(weights[i].rows, weights[i].columns);
        }
    }

    private void initializeCummulativeDerivatives() {
        for (int i = 1; i < innerLayers + 1; i++) {
            activationCummulativeDerivatives[i] = new Matrix(activations[i].rows);
            weightedSumCummulativeDerivatives[i] = new Matrix(weightedSums[i].rows);
            biasCummulativeDerivatives[i] = new Matrix(biases[i].rows);
        }
        for (int i = 1; i < innerLayers; i++) {
            weightCummulativeDerivatives[i] = new Matrix(weights[i].rows, weights[i].columns);
        }
    }

    public Matrix calculate(Matrix input) throws IllegalArgumentException {
        if (input.columns != 1) {
            throw new IllegalArgumentException();
        }
        activations[0] = input;
        for (int i = 1; i < activations.length; i++) {
            weightedSums[i] = weights[i - 1].multiply(activations[i - 1]).add(biases[i]);
            activations[i] = weightedSums[i].applyFunction(sigmoid);
        }
        return activations[activations.length - 1].applyFunction(sigmoid);
    }

    public void showTest(Matrix input, Matrix output) throws IllegalArgumentException {
        if (input.columns != 1) {
            throw new IllegalArgumentException();
        }
        initializeDerivatives();
        Matrix result = this.calculate(input);
        activationDerivatives[innerLayers + 1] = output.add(result.multiply(-1.0)).multiply(2);
        for (int i = innerLayers + 1; i >= 1; i--) {
            calculateActivationDerivatives(i);
            calculateWeightedSumDerivatives(i);
            calculateWeightDerivatives(i);
            calculateBiasDerivatives(i);
        }
        numberOfTests++;
    }

    private void calculateActivationDerivatives(int layer) {
        if (layer < innerLayers + 1) {
            activationDerivatives[layer] = weightedSumDerivatives[layer + 1].transpose().multiply(weights[layer])
                    .transpose();
            activationCummulativeDerivatives[layer] = activationCummulativeDerivatives[layer]
                    .add(activationDerivatives[layer]);
        }
    }
    
    private void calculateWeightedSumDerivatives(int layer) {
        weightedSumDerivatives[layer] = activationDerivatives[layer]
                .weirdMultiply(weightedSums[layer].applyFunction(sigmoidDerivative));
        weightedSumCummulativeDerivatives[layer] = weightedSumCummulativeDerivatives[layer]
                .add(weightedSumDerivatives[layer]);
    }
    
    private void calculateWeightDerivatives(int layer) {
        weightDerivatives[layer] = weightedSumDerivatives[layer].multiply(activations[layer].transpose());
        weightCummulativeDerivatives[layer] = weightCummulativeDerivatives[layer].add(weightDerivatives[layer]);
    }
    
    private void calculateBiasDerivatives(int layer) {
        biasDerivatives[layer] = weightedSumDerivatives[layer];
        biasCummulativeDerivatives[layer] = biasCummulativeDerivatives[layer].add(biasDerivatives[layer]);
    }

    public void backpropagate() {
        if(numberOfTests == 0) return;
        for (int i = 1; i < innerLayers + 1; i++) {
            biases[i] = biases[i].add(biasCummulativeDerivatives[i].multiply(1 / (2 * numberOfTests)));
            biasCummulativeDerivatives[i] = biasCummulativeDerivatives[i].multiply(1 / 2);
        }
        for (int i = 1; i < innerLayers; i++) {
            weights[i] = weights[i].add(weightCummulativeDerivatives[i].multiply(1 / (2 * numberOfTests)));
            weightCummulativeDerivatives[i] = weightCummulativeDerivatives[i].multiply(1 / 2);
        }
        numberOfTests /= 2;
    }
    
    public void saveAs(String name) throws FileAlreadyExistsException, IOException {
        Path path = Paths.get(name);
        //if (Files.exists(path))
        //    throw new FileAlreadyExistsException(name);
        Files.createDirectory(path);
        saveMatrixes(path, "biases", biases);
    }
    
    private static void saveMatrixes(Path path, String matrixName, Matrix[] matrixes)throws IOException {
        path = path.resolve(matrixName);
        Files.createDirectory(path);
        Path currentPath = path;
        for (int i = 0; i < matrixes.length; i++){
            currentPath = path.resolve(Integer.toString(i) + ".txt");
            File file = currentPath.toFile();
            file.createNewFile();
            matrixes[i].printMatrix(file);
        }
    }
}