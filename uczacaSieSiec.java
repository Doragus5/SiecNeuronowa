import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

public class UczacaSieSiec {
    public static void main(String[] args) {
        double[][] tab1 = { { 2, 1 }, { 2, 3 }, { 3, 0 } };
        double[][] tab2 = { { 1, 0 }, { 2, 3 } };
        Matrix ma1 = new Matrix(tab1);
        Matrix ma2 = new Matrix(tab2);
        int[] il = {2, 3};
        Network network = new Network(il, 1, 1);
        Random random1 = new Random();
        Random random2 = new Random();
        System.out.println(random1.nextGaussian());
        System.out.println(random2.nextGaussian());
        Matrix ma3 = new Matrix(2, 3);
        Matrix ma4 = new Matrix(3, 4);
        Matrix ma5 = new Matrix(4);
        System.out.println();
        ma4.multiply(ma5).printMatrix();
        System.out.println();
        ma3.multiply(ma4).printMatrix();
        System.out.println();
        ma3.multiply(ma4).transpose().printMatrix();
        System.out.println();
        ma1.printMatrix();
        System.out.println();
        (ma1.multiply(1, ma2)).printMatrix();
        System.out.println();
        (ma1.multiply(1, ma2)).transpose().printMatrix();
        try {

            Path path = Paths.get("test");
            Path path2 = Paths.get("name");
            path = path.resolve(path2);
            network.saveAs("siec1");
            Files.createDirectories(path);

            System.out.println(path.toString() + " is created!");

        } catch (IOException e) {

            System.err.println("Failed to create directory!" + e.getMessage());

        }
    }

}
