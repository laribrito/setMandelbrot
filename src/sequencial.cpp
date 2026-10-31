#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>
#include <cstdlib>
#include <filesystem>

// ==============================================================================
// Identificador da Etapa do Projeto a que este código pertence
// Use 1 para Etapa 1 (data/etapa1/), 2 para Etapa 2 (data/etapa2/), etc.
// ==============================================================================
const int ETAPA = 2;

class Complex {
public:
    double real;
    double imag;
    const double EPSILON = 1e-9;

    Complex(double r = 0.0, double i = 0.0) : real(r), imag(i) {}

    Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }

    Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }

    Complex operator*(const Complex& other) const {
        return Complex(real * other.real - imag * other.imag,
                       real * other.imag + imag * other.real);
    }

    Complex& operator=(const Complex& other) {
        real = other.real;
        imag = other.imag;
        return *this;
    }

    Complex& operator=(double val) {
        real = val;
        imag = 0.0;
        return *this;
    }

    bool operator==(const Complex& other) const {
        return std::abs(real - other.real) < EPSILON && std::abs(imag - other.imag) < EPSILON;
    }

    bool operator==(double val) const {
        return std::abs(real - val) < EPSILON && std::abs(imag) < EPSILON;
    }

    // Retorna o quadrado do módulo (útil para otimização em cálculos de fractais)
    double magnitudeSquared() const {
        return real * real + imag * imag;
    }

    Complex point(double x, double y){
        real = x;
        imag = y;
        return *this;
    }

    void print() const {
        std::cout << real << (imag >= 0 ? "+" : "" ) << imag << "i" << std::endl;
    }
};

struct RGB {
    int r, g, b;
};

// Retorna a cor da paleta clássica baseada no valor t_base entre 0.0 e 1.0
RGB getPaletteColor(double t_base) {
    double t = std::cbrt(t_base);
    int r = static_cast<int>(9   * (1-t) * t*t*t        * 255);
    int g = static_cast<int>(15  * (1-t)*(1-t) * t*t    * 255);
    int b = static_cast<int>(8.5 * (1-t)*(1-t)*(1-t)*t  * 255);
    return {r, g, b};
}

int main() {
    // Marca o início da ação completa
    auto totalStart = std::chrono::high_resolution_clock::now();

    // -------------------------------------------------------------
    // Cálculo
    // -------------------------------------------------------------

    // Parâmetros padrão
    int WIDTH = 4096, HEIGHT = 4096, MAX_ITER = 1000;
    double RE_MIN = -2.0, RE_MAX = 1.0, IM_MIN = -1.5, IM_MAX = 1.5;
    int etapaAtual = ETAPA;

    std::ifstream inFile("in.txt");
    if (inFile.is_open()) {
        std::string line;
        while (std::getline(inFile, line)) {
            // Ignora linhas vazias ou comentários (caso queira adicionar no futuro)
            if (line.empty() || line[0] == '#') continue; 
            
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);
                
                if (key == "WIDTH") WIDTH = std::stoi(value);
                else if (key == "HEIGHT") HEIGHT = std::stoi(value);
                else if (key == "MAX_ITER") MAX_ITER = std::stoi(value);
                else if (key == "RE_MIN") RE_MIN = std::stod(value);
                else if (key == "RE_MAX") RE_MAX = std::stod(value);
                else if (key == "IM_MIN") IM_MIN = std::stod(value);
                else if (key == "IM_MAX") IM_MAX = std::stod(value);
                else if (key == "ETAPA")  etapaAtual = std::stoi(value);
            }
        }
        inFile.close();
    } else {
        std::cerr << "Aviso: Nao foi possivel abrir in.txt. Utilizando resolucao 4096 e params padroes." << std::endl;
    }

    double tamPixel_re = (RE_MAX - RE_MIN)/WIDTH,
           tamPixel_im = (IM_MAX - IM_MIN)/HEIGHT;
    int limHeight = HEIGHT / 2 + 1;
    
    // Alocamos apenas a metade superior para economizar memória e evitar estouro de pilha
    std::vector<int> temps(limHeight * WIDTH, 0);

    // Começa a marcar o tempo de puro processamento
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < WIDTH; i++) {
        double x = RE_MIN + i * tamPixel_re;
        for (int j = 0; j < limHeight; j++) {
            double y = IM_MIN + j * tamPixel_im;
            Complex c(x, y);
            Complex z = 0;
            int temp = 0;
            
            // Loop de escape de Mandelbrot
            while (z.magnitudeSquared() < 4.0 && temp < MAX_ITER) {
                z = z * z + c;
                temp++;
            }
            
            temps[j * WIDTH + i] = temp;
        }
    }

    // Termina de marcar o tempo
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Garante caminho padronizado da base de dados com base na etapa configurada
    std::string etapaFolder = "data/etapa" + std::to_string(etapaAtual);
    std::filesystem::create_directories(etapaFolder);
    std::string csvPath = etapaFolder + "/dateTimeExecution.csv";

    // Determina o número da execução atual e se o CSV já possui o cabeçalho com TempoTotal
    int executionNum = 1;
    bool isNewFile = true;
    bool hasTempoTotal = false;
    std::vector<std::string> existingLines;

    {
        std::ifstream checkFile(csvPath);
        if (checkFile.is_open()) {
            std::string line;
            while (std::getline(checkFile, line)) {
                if (!line.empty()) {
                    existingLines.push_back(line);
                }
            }
            checkFile.close();

            if (!existingLines.empty()) {
                isNewFile = false;
                executionNum = static_cast<int>(existingLines.size()) + 1;
                if (existingLines[0].find("TempoTotal") != std::string::npos) {
                    hasTempoTotal = true;
                }
            }
        }
    }

    // Se o arquivo já existia mas não tinha a coluna TempoTotal no cabeçalho, migra
    if (!isNewFile && !hasTempoTotal) {
        std::ofstream rewriteFile(csvPath, std::ios::trunc);
        if (rewriteFile.is_open()) {
            rewriteFile << "DataHora,TempoGasto,TempoTotal,WIDTH,HEIGHT,MAX_ITER,RE_MIN,RE_MAX,IM_MIN,IM_MAX,Machine,Code,Schedule,ChunkSize,Threads\n";
            for (size_t k = 1; k < existingLines.size(); ++k) {
                std::string oldLine = existingLines[k];
                size_t firstComma = oldLine.find(',');
                if (firstComma != std::string::npos) {
                    size_t secondComma = oldLine.find(',', firstComma + 1);
                    if (secondComma != std::string::npos) {
                        oldLine.insert(secondComma + 1, "N/A,");
                    }
                }
                rewriteFile << oldLine << "\n";
            }
            rewriteFile.close();
        }
    }

    // Garante que a pasta out exista
    std::filesystem::create_directories("out");

    // Grava a tabela de inteiros para validação
    std::string tableFilename = "out/mandelbrot_sequencial_" + std::to_string(executionNum) + "_table.ppm";
    std::ofstream tableFile(tableFilename);
    if (tableFile.is_open()) {
        for (int j = 0; j < HEIGHT; ++j) {
            int targetRow = (j < limHeight) ? j : (HEIGHT - j);
            for (int i = 0; i < WIDTH; ++i) {
                tableFile << temps[targetRow * WIDTH + i] << (i == WIDTH - 1 ? "" : " ");
            }
            tableFile << "\n";
        }
        tableFile.close();
        std::cout << "Tabela de iteracoes salva com sucesso em '" << tableFilename << "'\n";
    } else {
        std::cerr << "Erro ao abrir " << tableFilename << " para escrita.\n";
    }

    // Grava a matriz em formato PPM P3
    std::string ppmFilename = "out/mandelbrot_sequencial_" + std::to_string(executionNum) + "_img.ppm";
    std::ofstream ppmFile(ppmFilename);
    if (ppmFile.is_open()) {
        int blackLineWidth = static_cast<int>(WIDTH * 0.01); // Linha preta 1% da largura
        int gradientWidth = static_cast<int>(WIDTH * 0.10); // Gradiente 10% da largura
        int ppmWidth = WIDTH + blackLineWidth + gradientWidth;
        
        ppmFile << "P3\n" << ppmWidth << " " << HEIGHT << "\n255\n";
        for (int j = 0; j < HEIGHT; ++j) {
            int targetRow = (j < limHeight) ? j : (HEIGHT - j);
            for (int i = 0; i < WIDTH; ++i) {
                int val = temps[targetRow * WIDTH + i];
                if (val >= MAX_ITER) {
                    ppmFile << "0 0 0 "; // Preto (pontos dentro do conjunto)
                } else {
                    RGB color = getPaletteColor(static_cast<double>(val) / MAX_ITER);
                    ppmFile << color.r << " " << color.g << " " << color.b << " ";
                }
            }
            
            // Linha preta separadora
            for (int i = 0; i < blackLineWidth; ++i) {
                ppmFile << "0 0 0 ";
            }
            
            // Gradiente da paleta - De baixo (t_base=0) para cima (t_base=1)
            double t_grad = static_cast<double>(HEIGHT - 1 - j) / (HEIGHT - 1);
            RGB gradColor = getPaletteColor(t_grad);
            
            for (int i = 0; i < gradientWidth; ++i) {
                ppmFile << gradColor.r << " " << gradColor.g << " " << gradColor.b << " ";
            }
            
            ppmFile << "\n";
        }
        ppmFile.close();
        std::cout << "Imagem em escala de cinza salva com sucesso em '" << ppmFilename << "'\n";
    } else {
        std::cerr << "Erro ao abrir " << ppmFilename << " para escrita.\n";
    }

    // Termina de marcar o tempo da ação completa
    auto totalEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> totalElapsed = totalEnd - totalStart;

    // Obtém a data e hora atual do sistema
    auto now = std::chrono::system_clock::now();
    std::time_t current_time = std::chrono::system_clock::to_time_t(now);

    // Salva o resultado adicionando (append) no arquivo CSV
    std::ofstream csvFile(csvPath, std::ios::app);
    if (csvFile.is_open()) {
        const char* machineName = std::getenv("COMPUTERNAME");
        if (!machineName) machineName = std::getenv("HOSTNAME");
        if (!machineName) machineName = "Unknown";

        if (isNewFile) {
            csvFile << "DataHora,TempoGasto,TempoTotal,WIDTH,HEIGHT,MAX_ITER,RE_MIN,RE_MAX,IM_MIN,IM_MAX,Machine,Code,Schedule,ChunkSize,Threads\n";
        }
        
        // Formato: YYYY-MM-DD HH:MM:SS,TempoGasto,TempoTotal,WIDTH,HEIGHT,MAX_ITER,RE_MIN,RE_MAX,IM_MIN,IM_MAX,Machine,Code,Schedule,ChunkSize,Threads
        csvFile << std::put_time(std::localtime(&current_time), "%Y-%m-%d %H:%M:%S") 
                << "," << elapsed.count()
                << "," << totalElapsed.count()
                << "," << WIDTH
                << "," << HEIGHT
                << "," << MAX_ITER
                << "," << RE_MIN
                << "," << RE_MAX
                << "," << IM_MIN
                << "," << IM_MAX 
                << "," << machineName 
                << ",sequencial,N/A,N/A,1\n";
        csvFile.close();
        
        std::cout << "Execução finalizada!\n";
        std::cout << "Tempo gasto (puro processamento): " << elapsed.count() << " segundos.\n";
        std::cout << "Tempo total (acao completa): " << totalElapsed.count() << " segundos.\n";
        std::cout << "Registro salvo em '" << csvPath << "'\n";
    } else {
        std::cerr << "Erro ao abrir " << csvPath << " para escrita.\n";
    }

    return 0;
}
