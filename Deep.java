
import java.util.Random;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.io.IOException;
import java.util.Scanner;
//import java.time.*;

abstract class Neuron implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = 1L;
    protected double Activation;

    Neuron() {

    }

    public double GetActivation() {
        return Activation;
    }

    protected double Sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    protected double SigmoidDerivative(double x) {
        return Sigmoid(x) * (1.0 - Sigmoid(x));
    }
}

class InputNeuron extends Neuron {
    /**
     *
     */
    private static final long serialVersionUID = 1L;

    InputNeuron(double Value) {
        super();
        Activation = Value;
    }

    InputNeuron() {
        super();
    }

    public void SetActivation(double Value) {
        Activation = Value;
    }
}

abstract class CalculatingNeuron extends Neuron {
    /**
     *
     */
    private static final long serialVersionUID = 1L;
    protected Neuron[] PreviousLayer;
    protected double[] Weights;
    protected double Bias;
    protected double WeightedSum;
    protected double[] CollectiveWeightsDerivatives;
    protected double[] WeightsDerivatives;
    protected double CollectiveBiasDerivative;
    protected double BiasDerivative;
    protected double ActivationDerivative;
    protected double WeightedSumDerivative;

    protected int NumberInLayer;

    CalculatingNeuron(Neuron[] previousLayer, int n, int LayerSize) {
        super();
        NumberInLayer = n;
        PreviousLayer = previousLayer;
        Weights = new double[PreviousLayer.length];
        Bias = Math.random() * 20.0 - 10.0;
        if (n == LayerSize - 1) {
            Bias = -100.0;
        }
        CollectiveWeightsDerivatives = new double[Weights.length];
        WeightsDerivatives = new double[Weights.length];
        Random random = new Random();
        for (int i = 0; i < Weights.length; i++) {
            Weights[i] = random.nextGaussian() * Math.sqrt(2.0 / (PreviousLayer.length + LayerSize));
        }
        CollectiveBiasDerivative = 0.0;

        BiasDerivative = 1.0;
    }

    public void InitializeWeights(int LayerSize) {

    }

    public void CalculateActivation() {
        WeightedSum = 0;
        for (int i = 0; i < Weights.length; i++) {
            WeightedSum += Weights[i] * PreviousLayer[i].GetActivation();
        }
        Activation = Sigmoid(WeightedSum);
        if (Activation > 1.0 || Activation < 0.0) {
            System.out.println("Babol " + Activation);
        }
    }

    public void CalculateDerivatives() {

        CalculateActivationDerivative();
        CalculateWeightedSumDerivative();
        CalculateWeightsDerivatives();
        CalculateBiasDerivative();
    }

    protected abstract void CalculateActivationDerivative();

    protected void CalculateWeightedSumDerivative() {
        WeightedSumDerivative = SigmoidDerivative(WeightedSum) * ActivationDerivative;
    }

    protected void CalculateWeightsDerivatives() {
        for (int i = 0; i < Weights.length; i++) {
            WeightsDerivatives[i] = PreviousLayer[i].GetActivation() * WeightedSumDerivative;
            CollectiveWeightsDerivatives[i] += WeightsDerivatives[i];
        }
    }

    protected void CalculateBiasDerivative() {
        BiasDerivative = WeightedSumDerivative;
        CollectiveBiasDerivative += BiasDerivative;
    }

    public void Learn(int NumberOfTests) {
        if (NumberOfTests == 0) {
            return;
        }
        for (int i = 0; i < Weights.length; i++) {
            Weights[i] -= 0.01 * CollectiveWeightsDerivatives[i] / NumberOfTests;
            CollectiveWeightsDerivatives[i] = 0.0;
        }
        Bias -= 0.001 * CollectiveBiasDerivative / NumberOfTests;
        CollectiveBiasDerivative = 0.0;
        NumberOfTests = 0;
    }
}

class OutputNeuron extends CalculatingNeuron {
    /**
     *
     */
    private static final long serialVersionUID = 1L;
    protected double ExpectedValue;

    OutputNeuron(Neuron[] previousLayer, int n, int LayerSize) {
        super(previousLayer, n, LayerSize);
    }

    public void SetExpectedValue(double x) {
        ExpectedValue = x;
    }

    protected void CalculateActivationDerivative() {
        ActivationDerivative = 2 * (Activation - ExpectedValue);
    }
}

class HiddenNeuron extends CalculatingNeuron {
    /**
     *
     */
    private static final long serialVersionUID = 1L;
    protected CalculatingNeuron[] NextLayer;

    HiddenNeuron(Neuron[] previousLayer, CalculatingNeuron[] nextLayer, int n, int LayerSize) {
        super(previousLayer, n, LayerSize);
        NextLayer = nextLayer;
    }

    protected void CalculateActivationDerivative() {
        ActivationDerivative = 0.0;
        for (int i = 0; i < NextLayer.length; i++) {
            ActivationDerivative += NextLayer[i].Weights[NumberInLayer] * NextLayer[i].WeightedSumDerivative;
        }
    }
}

class NeuronNetwork implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = 1L;
    private InputNeuron[] InputNeurons;
    private HiddenNeuron[][] HiddenNeurons;
    private OutputNeuron[] OutputNeurons;
    private double[] CollectiveCost;
    public int NumberOfTests;

    NeuronNetwork(int in, int hn, int hm, int on) {
        InputNeurons = new InputNeuron[in];
        HiddenNeurons = new HiddenNeuron[hn][hm];
        OutputNeurons = new OutputNeuron[on];
        CollectiveCost = new double[on];
        NumberOfTests = 0;
        for (int i = 0; i < InputNeurons.length; i++) {
            InputNeurons[i] = new InputNeuron();
        }
        if (hn == 1) {
            for (int j = 0; j < HiddenNeurons[0].length; j++) {
                HiddenNeurons[0][j] = new HiddenNeuron(InputNeurons, OutputNeurons, j, hm);
            }
        } else {
            for (int j = 0; j < HiddenNeurons[0].length; j++) {
                HiddenNeurons[0][j] = new HiddenNeuron(InputNeurons, HiddenNeurons[1], j, hm);
            }
        }
        for (int i = 1; i < HiddenNeurons.length - 1; i++) {
            for (int j = 0; j < HiddenNeurons[0].length; j++) {
                HiddenNeurons[i][j] = new HiddenNeuron(HiddenNeurons[i - 1], HiddenNeurons[i + 1], j, hm);
            }
        }
        for (int j = 0; j < HiddenNeurons[0].length && hn > 1; j++) {
            HiddenNeurons[HiddenNeurons.length - 1][j] = new HiddenNeuron(HiddenNeurons[HiddenNeurons.length - 2],
                    OutputNeurons, j, hm);
        }
        for (int i = 0; i < OutputNeurons.length; i++) {
            OutputNeurons[i] = new OutputNeuron(HiddenNeurons[HiddenNeurons.length - 1], i, hm);
            CollectiveCost[i] = 0.0;
        }
    }

    public double[] Calculate(double[] Input) {
        double[] Results = new double[OutputNeurons.length];
        for (int i = 0; i < InputNeurons.length; i++) {
            InputNeurons[i].SetActivation(Input[i]);
        }
        for (int i = 0; i < HiddenNeurons.length; i++) {
            for (int j = 0; j < HiddenNeurons[0].length; j++) {
                HiddenNeurons[i][j].CalculateActivation();
            }
        }
        for (int j = 0; j < OutputNeurons.length; j++) {
            OutputNeurons[j].CalculateActivation();
            Results[j] = OutputNeurons[j].GetActivation();
        }
        return Results;
    }

    public void ShowTest(double[] Input, double[] Output) {
        NumberOfTests++;
        for (int i = 0; i < InputNeurons.length; i++) {
            InputNeurons[i].SetActivation(Input[i]);
        }
        for (int i = 0; i < HiddenNeurons.length; i++) {
            for (int j = 0; j < HiddenNeurons[0].length; j++) {
                HiddenNeurons[i][j].CalculateActivation();
            }
        }
        for (int j = 0; j < OutputNeurons.length; j++) {
            OutputNeurons[j].CalculateActivation();
            OutputNeurons[j].SetExpectedValue(Output[j]);
            CollectiveCost[j] += (OutputNeurons[j].GetActivation() - Output[j])
                    * (OutputNeurons[j].GetActivation() - Output[j]);
            OutputNeurons[j].CalculateDerivatives();
        }
        for (int i = HiddenNeurons.length - 1; i >= 0; i--) {
            for (int j = 0; j < HiddenNeurons[0].length; j++) {
                HiddenNeurons[i][j].CalculateDerivatives();
            }
        }
    }

    public double[] Learn() {

        for (int i = 0; i < HiddenNeurons.length; i++) {
            for (int j = 0; j < HiddenNeurons[0].length; j++) {
                HiddenNeurons[i][j].Learn(NumberOfTests);
            }
        }
        for (int j = 0; j < OutputNeurons.length; j++) {
            OutputNeurons[j].Learn(NumberOfTests);
            CollectiveCost[j] /= NumberOfTests;
        }
        double[] Results = CollectiveCost;
        CollectiveCost = new double[CollectiveCost.length];
        for (int i = 0; i < CollectiveCost.length; i++) {
            CollectiveCost[i] = 0.0;
        }
        NumberOfTests = 0;
        return Results;
    }

    public double[] Random() {
        double[] Output = new double[OutputNeurons.length];
        Output[72] = 0.5;
        Output[(int) (Math.random() * 72)] = 1.0;
        return Output;
    }

    public double[] Bot(double[] GameState) {
        int[] Dice = new int[6];
        double[] Output = new double[OutputNeurons.length];
        Output[72] = 0.5;
        for (int i = 0; i < 36; i++) {
            if (GameState[i] == 1.0) {
                Dice[i % 6]++;
            }
        }
        int LastMove = 108;
        while (LastMove >= 36 && GameState[LastMove] < 0.5) {
            LastMove--;
        }
        LastMove -= 36;

        int Missing = 0;
        if (LastMove < 0) {
            Output[5] = 1.0;
            return Output;
        }
        Missing = LastMove / 6 + 1 - Dice[LastMove % 6];
        if (Missing > 2) {
            return Output;
        }
        int CurrentMissing = 3;
        while (LastMove < 72 && CurrentMissing > 2) {
            LastMove++;
            CurrentMissing = LastMove / 6 + 1 - Dice[LastMove % 6];
        }
        Output[LastMove] = 1.0;
        return Output;
    }
}

class DiceGame {

    DiceGame() {

    }

    public boolean Play(NeuronNetwork NN, boolean T) {
        double[][] GameState = new double[2][109];
        int[] MoveHistory = new int[73];
        int DummyDice;
        int[] NrOfDice = new int[6];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 6; j++) {
                DummyDice = (int) Math.floor(Math.random() * 6);
                NrOfDice[DummyDice]++;
                GameState[i][j * 6 + DummyDice] = 1.0;
            }
        }
        int FirstPossibleMove = 0;
        int LastPlayer = -1;
        double[] Move;
        double MinValue;
        int MinPlace;
        int WhoFirst = (int) (Math.random() * 2);
        while (FirstPossibleMove < 73) {
            LastPlayer++;
            LastPlayer %= 2;
            if (T && LastPlayer == WhoFirst) {
                Move = NN.Calculate(GameState[LastPlayer]);
            } else {
                Move = NN.Bot(GameState[LastPlayer]);
            }
            MinValue = Move[FirstPossibleMove];
            MinPlace = FirstPossibleMove;
            for (int i = FirstPossibleMove + 1; i < 73; i++) {
                if (Move[i] > MinValue && (FirstPossibleMove != 0 || i != 72)) {
                    MinPlace = i;
                    MinValue = Move[i];
                }
            }
            for (int i = 0; i < 2; i++) {
                GameState[i][MinPlace + 36] = 1.0;
                MoveHistory[MinPlace] = 1;
            }
            FirstPossibleMove = MinPlace + 1;
        }
        int LastMove = FindLastMove(MoveHistory);
        int Winner;
        double[] ExpectedOutput = new double[73];
        if (NrOfDice[LastMove % 6] > LastMove / 6) {
            Winner = (LastPlayer + 1) % 2;
            GameState[Winner][108] = 0.0;
            MoveHistory[72] = 0;
        } else {
            Winner = LastPlayer;
            GameState[Winner][108] = 0.0;
            MoveHistory[72] = 0;
            ExpectedOutput[72] = 1.0;
            NN.ShowTest(GameState[Winner], ExpectedOutput);

            ExpectedOutput[72] = 0.0;
        }
        boolean wygrany = true;
        if (Winner == WhoFirst) {
            wygrany = false;
        }
        LastPlayer = (LastPlayer + 2 - 1) % 2;
        while (LastMove >= 0) {
            MoveHistory[LastMove] = 0;
            GameState[Winner][LastMove] = 0.0;
            if (LastPlayer == Winner) {
                ExpectedOutput[LastMove] = 1.0;
            }
            NN.ShowTest(GameState[Winner], ExpectedOutput);
            ExpectedOutput[LastMove] = 0.0;
            LastPlayer = (LastPlayer + 2 - 1) % 2;
            LastMove = FindLastMove(MoveHistory);
        }
        return wygrany;
    }

    public boolean PlayVsAI(NeuronNetwork NN, Scanner scanner) {
        double[][] GameState = new double[2][109];
        int[] MoveHistory = new int[73];
        int DummyDice;
        int[] NrOfDice = new int[6];
        int WhoFirst = (int) (Math.random() * 2);
        for (int i = 0; i < 2; i++) {
            if (i == 1)
                System.out.println("Kosci gracza " + i);
            for (int j = 0; j < 6; j++) {

                DummyDice = (int) Math.floor(Math.random() * 6);
                if (i == 1)
                    System.out.print(DummyDice + 1 + " ");
                NrOfDice[DummyDice]++;
                GameState[i][j * 6 + DummyDice] = 1.0;
            }
            System.out.print("\n");
        }
        for (int j = 0; j < 6; j++) {
            // System.out.println(j + 1 + ": " + NrOfDice[j]);
        }

        int FirstPossibleMove = 0;
        int LastPlayer = -1;
        double[] Move;
        double MinValue;
        int MinPlace;
        int x, y;
        while (FirstPossibleMove < 73) {
            LastPlayer++;
            LastPlayer %= 2;
            if (LastPlayer == WhoFirst) {
                Move = NN.Calculate(GameState[LastPlayer]);
                MinValue = Move[FirstPossibleMove];
                MinPlace = FirstPossibleMove;
                for (int i = FirstPossibleMove + 1; i < 73; i++) {
                    if (Move[i] > MinValue && (FirstPossibleMove != 0 || i != 72)) {
                        MinPlace = i;
                        MinValue = Move[i];
                    }
                }
                for (int i = 0; i < 2; i++) {
                    GameState[i][MinPlace + 36] = 1.0;
                    MoveHistory[MinPlace] = 1;
                }
                System.out.println("Ruch komputera: " + (MinPlace / 6 + 1) + " " + (MinPlace % 6 + 1));
                FirstPossibleMove = MinPlace + 1;
            } else {
                System.out.println("twoj ruch");
                x = scanner.nextInt();
                y = scanner.nextInt();
                x--;
                y--;
                MinPlace = x * 6 + y;
                while (MinPlace < FirstPossibleMove || MinPlace > 72 || y > 5) {
                    System.out.println("Wykonaj inny ruch");
                    x = scanner.nextInt();
                    y = scanner.nextInt();
                    x--;
                    y--;
                    MinPlace = x * 6 + y;
                }

                for (int i = 0; i < 2; i++) {
                    GameState[i][MinPlace + 36] = 1.0;
                    MoveHistory[MinPlace] = 1;
                }
                FirstPossibleMove = MinPlace + 1;
            }
        }
        int LastMove = FindLastMove(MoveHistory);
        int Winner;
        double[] ExpectedOutput = new double[73];
        if (NrOfDice[LastMove % 6] > LastMove / 6) {
            Winner = (LastPlayer + 1) % 2;
            GameState[Winner][108] = 0.0;
            MoveHistory[72] = 0;
        } else {
            Winner = LastPlayer;
            GameState[Winner][108] = 0.0;
            MoveHistory[72] = 0;
            ExpectedOutput[72] = 1.0;
            NN.ShowTest(GameState[Winner], ExpectedOutput);

            ExpectedOutput[72] = 0.0;
        }
        boolean wygrany = true;
        if (Winner == WhoFirst) {
            System.out.println("Komputer wygrał!");
            wygrany = false;
        } else {
            System.out.println("Wygrałeś!");
        }
        LastPlayer = (LastPlayer + 2 - 1) % 2;
        while (LastMove >= 0) {
            MoveHistory[LastMove] = 0;
            GameState[Winner][LastMove] = 0.0;
            if (LastPlayer == Winner) {
                ExpectedOutput[LastMove] = 1.0;

            }
            NN.ShowTest(GameState[Winner], ExpectedOutput);
            ExpectedOutput[LastMove] = 0.0;
            LastPlayer = (LastPlayer + 2 - 1) % 2;
            LastMove = FindLastMove(MoveHistory);
        }
        return wygrany;
    }

    private int FindLastMove(int[] MoveHistory) {
        int i = 71;
        while (i >= 0 && MoveHistory[i] == 0)
            i--;
        return i;
    }
}

public class Deep {
    public static void main(String[] args) {
        try {
            FileOutputStream fos = new FileOutputStream("Network.dat");
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            System.out.println("Zaczynam!");
            NeuronNetwork[] Sieci = new NeuronNetwork[2];
            Sieci[0] = new NeuronNetwork(109, 2, 100, 73);
            Sieci[1] = new NeuronNetwork(109, 2, 73, 73);
            DiceGame Gra = new DiceGame();
            double[] Koszty;
            double MaxKoszt;
            double AvgKoszt = 0;
            // String strDouble;
            int LiczbaGier = 1000;
            // Instant start = Instant.now();
            // Instant mid;
            int[] Testy = new int[2];
            int i = 1;
            while (Testy[0] < LiczbaGier * 2 / 5) {
                if (Gra.Play(Sieci[0], true)) {
                    Testy[1]++;
                } else {
                    Testy[0]++;
                }
                if (i % 1 == 0) {
                    // System.out.print("\033[H\033[2J");
                    // System.out.flush();
                    // mid = Instant.now();
                    // Duration timeElapsed = Duration.between(start, mid);
                    // System.out.println("Time left: " + (timeElapsed.toSeconds() *
                    // ((double)LiczbaGier / (double)i) - timeElapsed.toSeconds()) + " seconds");
                    // System.out.println(((double)i / (double)(LiczbaGier / 100)) + "%");
                    // System.out.println("Wygrany sieci: " + Testy[0] + " Wygrane bota: " +
                    // Testy[1]);
                    Koszty = Sieci[0].Learn();
                    MaxKoszt = Koszty[0];
                    AvgKoszt = 0;
                    for (int j = 0; j < Koszty.length; j++) {
                        AvgKoszt += Koszty[j];
                        if (Koszty[j] > MaxKoszt) {
                            MaxKoszt = Koszty[j];
                        }
                    }

                    // strDouble = String.format("%.8f", MaxKoszt);
                    // System.out.println("Max koszt: " + Math.sqrt(MaxKoszt));
                     System.out.println("Sredni koszt: " + Math.sqrt(AvgKoszt / Koszty.length));
                    // System.out.println("Koszty: " + Arrays.toString(Koszty));
                }
                if (i % LiczbaGier == 0) {
                    System.out.print("\033[H\033[2J");
                    System.out.flush();
                    System.out.println("Wygrane sieci: " + Testy[0] + " Wygrane bota: " + Testy[1]);
                    Testy[0] = 0;
                    Testy[1] = 0;
                }
                i++;
            }
            oos.writeObject(Sieci);
            oos.close();
            System.out.println("Yay!");
            Scanner scanner = new Scanner(System.in);
            int Komp = 0;
            int gracz = 0;
            for (i = 0; i < 1000; i++) {
                if (Gra.PlayVsAI(Sieci[0], scanner)) {
                    gracz++;
                } else {
                    Komp++;
                }
                if (i % 3 == 0) {
                    Sieci[0].Learn();
                }
                System.out.println("Komputer: " + Komp + " Gracz: " + gracz);
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
}