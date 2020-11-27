import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.Arrays;
import java.util.Scanner;

public class Matrix {
    public final int columns, rows;
    protected double matrix[][];
    protected double multiplier;
    protected boolean isIdentity = false;

    Matrix(double[][] matrix) {
        multiplier = 1.0;
        rows = matrix.length;
        columns = matrix[0].length;
        copyFromArray(matrix);
    }
    
    Matrix(int r, int c) {
        multiplier = 1.0;
        this.rows = r;
        this.columns = c;
        matrix = new double[r][c];
    }

    Matrix(int r) {
        multiplier = 1.0;
        this.rows = r;
        this.columns = 1;
        matrix = new double[r][columns];
    }

    Matrix(Matrix ma) {
        multiplier = ma.multiplier;
        columns = ma.columns;
        rows = ma.rows;
        copyFromArray(ma.matrix);
    }

    Matrix(File file) throws FileNotFoundException{
        Scanner scanner = new Scanner(file);
        int dummyr;
        if (!scanner.hasNextInt()) {
            columns = -1;
            rows = -1;
            multiplier = 0.0;
            matrix = new double[0][0];
            scanner.close();
            return;
        } else {
            columns = scanner.nextInt();
            if (!scanner.hasNextInt()) {
                rows = -1;
                multiplier = 0.0;
                matrix = new double[0][0];
                scanner.close();
                return;
            } else {
                dummyr = scanner.nextInt();
            }
        }
        multiplier = 1.0;
        matrix = new double[dummyr][columns];
        int i = 0;
        while (scanner.hasNextDouble()) {
            matrix[i / dummyr][i % dummyr] = scanner.nextDouble();
            i++;
        }
        if (i < dummyr * columns) {
            rows = -1;
            multiplier = 0.0;
            matrix = new double[0][0];
            scanner.close();
            return;
        } else {
            rows = dummyr;
        }
        scanner.close();
    }

    Matrix() {
        columns = 0;
        rows = 0;
        multiplier = 1;
        isIdentity = true;
    }

    public Matrix multiply(double multiplier, Matrix ma) {
        return this.multiply(ma.multiply(multiplier));
    }

    public Matrix multiply(Matrix ma) {
        if (columns != ma.rows) {
            throw new IllegalArgumentException();
        }
        double[][] zawartosc = new double[rows][ma.columns];
        for (int i = 0; i < ma.columns; i++) {
            for (int j = 0; j < rows; j++) {
                for (int k = 0; k < columns; k++) {
                    zawartosc[j][i] += matrix[j][k] * ma.matrix[k][i];
                }

            }
        }
        Matrix result = new Matrix(zawartosc);
        result.multiplier *= multiplier * ma.multiplier;
        return result;
    }

    public Matrix add(Matrix ma) {
        if (columns != ma.columns || rows != ma.rows) {
            throw new IllegalArgumentException();
        }
        double[][] zawartosc = new double[rows][ma.columns];
        for (int i = 0; i < columns; i++) {
            for (int j = 0; j < rows; j++) {
                zawartosc[j][i] = multiplier * matrix[j][i] + ma.multiplier * ma.matrix[j][i];
            }
        }
        return new Matrix(zawartosc);
    }

    public Matrix multiply(double multiplier) {
        if (multiplier == 0) {
            return new Matrix(new double[rows][columns]);
        }
        Matrix result;
        result = new Matrix(this);
        result.multiplier = multiplier * result.multiplier;
        return result;
    }

    protected void copyFromArray(double[][] matrix) {
        this.matrix = new double[rows][];
        for (int i = 0; i < rows; i++) {
            this.matrix[i] = Arrays.copyOf(matrix[i], columns);
        }
    }

    public void printMatrix() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                if (isIdentity) {
                    System.out.print((i == j) + " ");
                } else {
                    System.out.print(matrix[i][j] * multiplier + " ");
                }
            }
            System.out.print("\n");
        }
    }

    public Matrix applyFunction(DoubleUnaryOperator function) {
        Matrix resultMatrix = new Matrix(this);
        resultMatrix.functionTheMatrix(function);
        return resultMatrix;

    }
    
    private void functionTheMatrix(DoubleUnaryOperator func) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] = func.applyAsDouble(matrix[i][j]);
            }
        }
    }

    public Matrix applyFunction(DoubleBinaryOperator function) {
        Matrix resultMatrix = new Matrix(this);
        resultMatrix.functionTheMatrix(function);
        return resultMatrix;
    }

    private void functionTheMatrix(DoubleBinaryOperator func) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] = func.applyAsDouble(columns, rows);
            }
        }
    }

    public Matrix weirdMultiply(Matrix ma) throws IllegalArgumentException {
        if (1 != ma.columns || rows != ma.rows) {
            throw new IllegalArgumentException();
        }
        Matrix resultMatrix = new Matrix(this);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                resultMatrix.matrix[i][j] *= ma.matrix[i][1];
            }
        }
        return resultMatrix;
    }

    public Matrix extendNTimes(int n) {
        Matrix resultMatrix = new Matrix(rows, columns * n);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns * n; j++) {
                resultMatrix.matrix[i][j] = matrix[i][j % columns];
            }
        }
        return resultMatrix;
    }

    public double getValue(int x, int y) {
        return matrix[y][x];
    }

    public Matrix transpose() {
        Matrix transposed = new Matrix(columns, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                transposed.matrix[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }


    public void printMatrix(File file) {
        try{
            FileWriter fileWriter = new FileWriter(file);
            fileWriter.write(columns + " " + rows + "\n");
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                   fileWriter.write(matrix[i][j] * multiplier + " ");
                }
                fileWriter.write("\n");
            }
            fileWriter.close();
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }
}
