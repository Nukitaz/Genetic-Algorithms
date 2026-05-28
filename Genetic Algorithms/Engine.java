package genetic_benetton;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

/*
Kevin Jair Nuno Trujillo
Algoritmo Genetico
Fecha: 26/05/2026
*/

/*
  Engine.java
  Contiene la logica evolutiva del Algoritmo Genético:
  Population — gestiona los N cromosomas candidatos
  GeneticAlgorithm — orquesta el bucle completo hasta CoD > umbral
*/

/*
Population
Almacena y gestiona los cromosomas de una generacion.
Implementa los dos mecanismos de selección del documento:
  Primer padre  → recorrido secuencial con nr < crossoverRate
  Segundo padre → ruleta proporcional al fitness (R^2)    
*/

class Population {

    private Chromosome[] chromosomes;
    private final int    size;
    private final double minVal;
    private final double maxVal;

    private static final Random rng = new Random();

    public Population(int size, double minVal, double maxVal) {
        this.size       = size;
        this.minVal     = minVal;
        this.maxVal     = maxVal;
        this.chromosomes = new Chromosome[size];
    }

    //inicializacion aleatoria (0)
    public void initialize() {
        for (int i = 0; i < size; i++) {
            chromosomes[i] = Chromosome.random(minVal, maxVal);
        }
    }

    //reemplaza la poblacion con la nueva generación
    public void setChromosomes(Chromosome[] newGen) {
        this.chromosomes = newGen;
    }

    //seleccion del 1er padre
    //recorre la poblacion en orden; elige el primero donde nr <= crossoverRate
    public Chromosome selectFirstParent(double crossoverRate) {
        for (Chromosome c : chromosomes) {
            if (rng.nextDouble() <= crossoverRate) return c;
        }
        return chromosomes[0]; // fallback: siempre devuelve uno
    }

    //seleccion del 2do padre — Ruleta (fitness proporcional)
    //cromosomas con mayor R^2 tienen mayor probabilidad de ser elegidos
    public Chromosome selectSecondParent() {
        double total = 0;
        for (Chromosome c : chromosomes) total += c.getFitness();

        if (total == 0) return chromosomes[rng.nextInt(size)]; // edge case

        double spin = rng.nextDouble() * total;
        double acum = 0;
        for (Chromosome c : chromosomes) {
            acum += c.getFitness();
            if (acum >= spin) return c;
        }
        return chromosomes[size - 1];
    }

    //retorna el cromosoma con mayor R^2 en la generacion actual
    public Chromosome getBest() {
        return Arrays.stream(chromosomes)
                .max(Comparator.comparingDouble(Chromosome::getFitness))
                .orElse(chromosomes[0]);
    }

    public Chromosome[] getChromosomes() { return chromosomes; }
    public int          getSize()        { return size; }
}


/* 
GeneticAlgorithm
    Orquesta el ciclo evolutivo completo siguiendo el flujo del documento:
        1. generar poblacion inicial
        2. evaluar con R^2
        3. bucle: seleccion → cruza → mutacion → evaluacion
        4. terminar si CoD ≥ 90 % o se alcanza el maximo de generaciones
*/
class GeneticAlgorithm {

    //Parametros del AG (segun el documento)
    private static final int    POPULATION_SIZE = 100;
    private static final double CROSSOVER_RATE  = 0.95;
    private static final double MUTATION_RATE   = 0.01;
    private static final double COD_THRESHOLD   = 0.90;   // R^2 > 90 %
    private static final int    MAX_GENERATIONS = 20_000;
    private static final double MIN_VAL         = -500.0;
    private static final double MAX_VAL         =  500.0;

    //componentes
    private final Population        population;
    private final FitnessFunction   fitnessFunction;
    private final CrossoverOperator crossover;
    private final MutationOperator  mutation;

    public GeneticAlgorithm(double[] x, double[] y) {
        this.population      = new Population(POPULATION_SIZE, MIN_VAL, MAX_VAL);
        this.fitnessFunction = new FitnessFunction(x, y);
        this.crossover       = new CrossoverOperator();
        this.mutation        = new MutationOperator(MUTATION_RATE, MIN_VAL, MAX_VAL);
    }

    //metodo principal: ejecuta el AG y retorna el mejor cromosoma encontrado
    public Chromosome run() {

        //paso 2 — Poblacion inicial aleatoria
        population.initialize();
        evaluateAll();

        printHeader();
        Chromosome best = population.getBest();
        printGen(0, best);

        //paso 4-6 — bucle evolutivo
        for (int gen = 1; gen <= MAX_GENERATIONS; gen++) {

            Chromosome[] nextGen = new Chromosome[POPULATION_SIZE];
            int i = 0;

            while (i < POPULATION_SIZE) {
                //paso 4a — seleccion padre 1 (secuencial con crossoverRate)
                Chromosome p1 = population.selectFirstParent(CROSSOVER_RATE);

                //paso 4b — seleccion padre 2 (ruleta)
                Chromosome p2 = population.selectSecondParent();

                //paso 4c — cruza one-point
                Chromosome[] hijos = crossover.crossover(p1, p2);

                //paso 5 — mutacion one-point
                mutation.mutate(hijos[0]);
                mutation.mutate(hijos[1]);

                //paso 6 — evaluar hijos
                fitnessFunction.evaluate(hijos[0]);
                fitnessFunction.evaluate(hijos[1]);

                nextGen[i] = hijos[0];
                if (i + 1 < POPULATION_SIZE) nextGen[i + 1] = hijos[1];
                i += 2;
            }

            population.setChromosomes(nextGen);
            best = population.getBest();

            if (gen % 500 == 0) printGen(gen, best);

            //paso 7 — condicion de parada: CoD > 90 %
            if (best.getFitness() >= COD_THRESHOLD) {
                System.out.printf("%n  Condicion alcanzada en generacion %d%n", gen);
                printGen(gen, best);
                return best;
            }
        }

        System.out.printf("%n  Maximo de generaciones (%d) alcanzado.%n", MAX_GENERATIONS);
        printGen(MAX_GENERATIONS, best);
        return best;
    }

    //evalua todos los cromosomas de la generacion actual
    private void evaluateAll() {
        for (Chromosome c : population.getChromosomes()) {
            fitnessFunction.evaluate(c);
        }
    }

    //salida por consola
    private void printHeader() {
        System.out.println("=".repeat(60));
        System.out.println("  Algoritmo Genetico — Regresion Lineal (Caso Benetton)");
        System.out.println("=".repeat(60));
        System.out.printf("  Poblacion: %d  |  Cruza: %.2f  |  Mutacion: %.2f  |  CoD > %.0f%%%n%n",
                POPULATION_SIZE, CROSSOVER_RATE, MUTATION_RATE, COD_THRESHOLD * 100);
    }

    private void printGen(int gen, Chromosome best) {
        System.out.printf("  Gen %6d -> %s%n", gen, best);
    }
}