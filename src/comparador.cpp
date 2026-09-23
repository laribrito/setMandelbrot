#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Uso: comparador.exe arquivo1.txt arquivo2.txt\n";
        return 1;
    }

    std::ifstream file1(argv[1]);
    std::ifstream file2(argv[2]);

    if (!file1.is_open()) {
        std::cerr << "Erro ao abrir " << argv[1] << "\n";
        return 1;
    }
    if (!file2.is_open()) {
        std::cerr << "Erro ao abrir " << argv[2] << "\n";
        return 1;
    }

    std::string line1, line2;
    int row = 1;
    long long diffCount = 0;
    const int MAX_PRINTS = 20;

    std::cout << "Comparando:\n";
    std::cout << "A) " << argv[1] << "\n";
    std::cout << "B) " << argv[2] << "\n\n";

    while (std::getline(file1, line1) && std::getline(file2, line2)) {
        std::stringstream ss1(line1);
        std::stringstream ss2(line2);
        int val1, val2;
        int col = 1;

        while (ss1 >> val1 && ss2 >> val2) {
            if (val1 != val2) {
                if (diffCount < MAX_PRINTS) {
                    std::cout << "Diferenca na Linha " << row << ", Coluna " << col 
                              << " -> A: " << val1 << " | B: " << val2 << "\n";
                }
                diffCount++;
            }
            col++;
        }
        row++;
    }

    if (diffCount == 0) {
        std::cout << "\nSucesso! As duas matrizes sao EXATAMENTE iguais.\n";
    } else {
        std::cout << "\nATENCAO: Foram encontradas " << diffCount << " diferencas no total!\n";
        if (diffCount > MAX_PRINTS) {
            std::cout << "(Mostrando apenas as primeiras " << MAX_PRINTS << ")\n";
        }
    }

    return 0;
}
