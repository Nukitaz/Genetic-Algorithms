package genetic_benetton;

import java.util.Random;

/*
Kevin Jair Nuno Trujillo
Algoritmo Genetico
Fecha: 26/05/2026
*/

/*
models.java
    contiene las clases de datos y operadores del Algoritmo Genetico:
    Chromosome        — solucion candidata [b0, b1]
    FitnessFunction   — evalua R^2 de un cromosoma
    CrossoverOperator — one-point crossover entre dos padres    
    MutationOperator  — one-point mutation sobre un hijo
*/


//chromosome
//representa una solucion candidata para y = b0 + b1·x
//sus "genes" son los coeficientes b0 y b1 (numeros reales)
class Chromosome {

    private double beta0;
    private double beta1;
    private double fitness; //R^2 asignado por FitnessFunction

    private static final Random rng = new Random();

    //constructor directo con valores conocidos
    public Chromosome(double beta0, double beta1) {
        this.beta0   = beta0;
        this.beta1   = beta1;
        this.fitness = 0.0;
    }

    //fabrica: genera un cromosoma aleatorio dentro del rango dado
    public static Chromosome random(double minVal, double maxVal) {
        double b0 = minVal + (maxVal - minVal) * rng.nextDouble();
        double b1 = minVal + (maxVal - minVal) * rng.nextDouble();
        return new Chromosome(b0, b1);
    }

    //acceso a genes por indice (0=b0, 1=b1)
    public int    length()              { return 2; }
    public double getGene(int i)        { return (i == 0) ? beta0 : beta1; }
    public void   setGene(int i, double v) { if (i == 0) beta0 = v; else beta1 = v; }

    //getters y setters
    public double getBeta0()            { return beta0; }
    public double getBeta1()            { return beta1; }
    public double getFitness()          { return fitness; }
    public void   setFitness(double f)  { this.fitness = f; }

    //copia profunda: evita aliasing al crear hijos
    public Chromosome copy() {
        Chromosome c = new Chromosome(this.beta0, this.beta1);
        c.fitness = this.fitness;
        return c;
    }

    @Override
    public String toString() {
        return String.format("Chromosome [b0 = %.4f, b1 = %.4f, R^2 = %.4f]",
                beta0, beta1, fitness);
    }
}


/*FitnessFunction
evalua la aptitud de un cromosoma mediante el Coeficiente de Determinacion:

--              sigma(yi - ŷi)²                             --
--     R^2 = 1 − ────────────    con  ŷi = b0 + b1·xi       --
--              sigma(yi − ȳ)²                              --

resultado clampado a [0, 1]
*/
class FitnessFunction {

    private final double[] x;      //variable independiente (Advertising)
    private final double[] y;      //variable dependiente   (Sales)
    private final double   yMean;  //precalculado para eficiencia

    public FitnessFunction(double[] x, double[] y) {
        this.x     = x;
        this.y     = y;
        this.yMean = mean(y);
    }

    /**
    calcula R^2 y lo asigna como fitness del cromosoma
    y retorna el valor de R^2 en [0, 1]
    */
    public double evaluate(Chromosome c) {
        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < x.length; i++) {
            double yHat = c.getBeta0() + c.getBeta1() * x[i];
            ssRes += Math.pow(y[i] - yHat,   2);
            ssTot += Math.pow(y[i] - yMean,  2);
        }
        double r2 = (ssTot == 0) ? 0 : 1.0 - ssRes / ssTot;
        r2 = Math.max(0, Math.min(1, r2));   //clamp
        c.setFitness(r2);
        return r2;
    }

    private double mean(double[] v) {
        double s = 0;
        for (double d : v) s += d;
        return s / v.length;
    }
}


/*CrossoverOperator  —  One-Point Crossover

1. genera nr "existente" [1, length) como punto de corte
2. los genes a partir de nr se intercambian entre los dos padres

ejemplo con length=2, nr=1:
padre1=[b0a, b1a]  padre2=[b0b, b1b]
hijo1 =[b0a, b1b]  hijo2 =[b0b, b1a]
*/
class CrossoverOperator {

    private final Random rng = new Random();

    /**
    aplica one-point crossover y retorna dos hijos
    no modifica a los padres originales
    */
    public Chromosome[] crossover(Chromosome p1, Chromosome p2) {
        int length     = p1.length();
        int crossPoint = 1 + rng.nextInt(Math.max(1, length - 1));

        Chromosome o1 = p1.copy();
        Chromosome o2 = p2.copy();

        for (int i = crossPoint; i < length; i++) {
            double tmp = o1.getGene(i);
            o1.setGene(i, o2.getGene(i));
            o2.setGene(i, tmp);
        }
        return new Chromosome[]{o1, o2};
    }
}


/* 
MutationOperator  —  One-Point Mutation

Para cada cromosoma:
1. genera nr "existente [0, 1]
2. si nr <= mutationRate, elige un gen aleatorio y lo reemplaza
*/
class MutationOperator {

    private final Random rng          = new Random();
    private final double mutationRate;
    private final double minVal;
    private final double maxVal;

    public MutationOperator(double mutationRate, double minVal, double maxVal) {
        this.mutationRate = mutationRate;
        this.minVal       = minVal;
        this.maxVal       = maxVal;
    }

    /**
    muta el cromosoma in-place si el numero aleatorio lo indica
    y retorna true si hubo mutacion
    */
    public boolean mutate(Chromosome c) {
        if (rng.nextDouble() <= mutationRate) {
            int    geneIdx  = rng.nextInt(c.length());
            double newValue = minVal + (maxVal - minVal) * rng.nextDouble();
            c.setGene(geneIdx, newValue);
            return true;
        }
        return false;
    }
}