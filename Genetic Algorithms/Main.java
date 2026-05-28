package genetic_benetton;

/*
Kevin Jair Nuno Trujillo
Algoritmo Genetico
Fecha: 26/05/2026
*/

// Main.java
// punto de entrada del programa

// carga el Dataset del Caso Benetton, ejecuta el Algoritmo Genetico y muestra los coeficientes encontrados junto con las predicciones

//modelo buscado:  y = b0 + b1·x
//x = Advertising (Millones de Euro)
//y = Sales       (Millones de Euro)

public class Main {

    // Dataset — Caso Benetton
    private static final double[] ADVERTISING = {23, 26, 30, 34, 43, 48, 52, 57, 58};
    private static final double[] SALES       = {651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518};

    public static void main(String[] args) {

        // 1. ejecutar el Algoritmo Genetico
        GeneticAlgorithm ga       = new GeneticAlgorithm(ADVERTISING, SALES);
        Chromosome       solution = ga.run();

        // 2. mostrar coeficientes encontrados
        printSolution(solution);

        // 3. mostrar tabla de predicciones vs valores reales
        printPredictions(solution);
    }

    //imprime el modelo resultante y sus metricas
    private static void printSolution(Chromosome c) {
        System.out.println();
        System.out.println("=".repeat(60));
        System.out.println("  RESULTADO FINAL");
        System.out.println("=".repeat(60));
        System.out.printf("  b0 (intercepto) = %10.4f%n",  c.getBeta0());
        System.out.printf("  b1 (pendiente)  = %10.4f%n",  c.getBeta1());
        System.out.printf("  R^2  (CoD)       = %10.4f  (%.2f%%)%n",
                c.getFitness(), c.getFitness() * 100);
        System.out.printf("%n  Modelo: y = %.4f + %.4f · x%n",
                c.getBeta0(), c.getBeta1());
    }

    //imprime tabla comparativa: valor real vs prediccion vs error
    private static void printPredictions(Chromosome c) {
        System.out.println();
        System.out.println("=".repeat(60));
        System.out.println("  PREDICCIONES vs VALORES REALES");
        System.out.println("=".repeat(60));
        System.out.printf("  %-6s  %-14s  %-14s  %-10s%n",
                "Advert.", "Ventas reales", "Ventas pred.", "Error");
        System.out.println("  " + "-".repeat(50));

        for (int i = 0; i < ADVERTISING.length; i++) {
            double pred  = c.getBeta0() + c.getBeta1() * ADVERTISING[i];
            double error = SALES[i] - pred;
            System.out.printf("  %-6.0f  %-14.0f  %-14.2f  %+.2f%n",
                    ADVERTISING[i], SALES[i], pred, error);
        }
        System.out.println();
    }
}