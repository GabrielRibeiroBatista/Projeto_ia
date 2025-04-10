# Projeto de IA de Classificação Binária

Este projeto implementa um sistema de Inteligência Artificial para classificação binária, utilizando bibliotecas modernas de Python como scikit-learn, pandas e numpy.

## Funcionalidades

- Geração ou carregamento de dados tabulares
- Pré-processamento e normalização de dados
- Treinamento de modelos de classificação (Random Forest ou Regressão Logística)
- Avaliação de desempenho com métricas como acurácia e relatório de classificação
- Salvamento e carregamento de modelos treinados
- Inferência em novos dados

## Estrutura do Projeto

```
projeto_ia/
├── data/                  # Diretório para armazenar datasets (opcional)
│   └── dataset.csv        # Dataset de exemplo (opcional)
├── models/                # Diretório para armazenar modelos treinados
│   └── modelo_treinado.pkl # Modelo salvo após treinamento
├── src/                   # Código-fonte do projeto
│   ├── preprocess.py      # Módulo de pré-processamento de dados
│   ├── train.py           # Módulo de treinamento de modelos
│   └── predict.py         # Módulo de predição com modelos treinados
├── main.py                # Script principal que integra todos os módulos
├── README.md              # Este arquivo
└── requirements.txt       # Dependências do projeto
```

## Requisitos

- Python 3.x
- Bibliotecas listadas em `requirements.txt`

## Instalação

1. Clone este repositório:
```
git clone https://github.com/seu-usuario/projeto-ia-classificacao.git
cd projeto-ia-classificacao
```

2. Crie um ambiente virtual (opcional, mas recomendado):
```
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. Instale as dependências:
```
pip install -r requirements.txt
```

## Uso

### Executar o fluxo completo (pré-processamento, treinamento e predição)

```
python main.py --modo completo --sintetico --amostras 1000
```

### Apenas treinar um modelo

```
python main.py --modo treinar --modelo random_forest --estimadores 100 --sintetico
```

### Apenas fazer predições com um modelo existente

```
python main.py --modo prever --arquivo-modelo models/modelo_treinado.pkl --dados-predicao data/novos_dados.csv
```

### Opções disponíveis

```
python main.py --help
```

Saída:
```
usage: main.py [-h] [--modo {completo,treinar,prever}] [--dados DADOS]
               [--sintetico] [--amostras AMOSTRAS]
               [--modelo {random_forest,logistic_regression}]
               [--estimadores ESTIMADORES] [--max-depth MAX_DEPTH]
               [--test-size TEST_SIZE] [--arquivo-modelo ARQUIVO_MODELO]
               [--dados-predicao DADOS_PREDICAO]

Sistema de IA para classificação binária

optional arguments:
  -h, --help            show this help message and exit
  --modo {completo,treinar,prever}
                        Modo de operação: completo, treinar ou prever (default: completo)
  --dados DADOS         Caminho para arquivo CSV com dados (opcional) (default: None)
  --sintetico           Usar dados sintéticos em vez de arquivo (default: False)
  --amostras AMOSTRAS   Número de amostras para dados sintéticos (default: 1000)
  --modelo {random_forest,logistic_regression}
                        Tipo de modelo a ser usado (default: random_forest)
  --estimadores ESTIMADORES
                        Número de estimadores para Random Forest (default: 100)
  --max-depth MAX_DEPTH
                        Profundidade máxima para Random Forest (default: None)
  --test-size TEST_SIZE
                        Proporção do conjunto de teste (default: 0.2)
  --arquivo-modelo ARQUIVO_MODELO
                        Caminho para o arquivo do modelo treinado (default: models/modelo_treinado.pkl)
  --dados-predicao DADOS_PREDICAO
                        Caminho para arquivo CSV com dados para predição (default: None)
```

## Exemplos de Uso

### Exemplo 1: Treinar com dados sintéticos e Random Forest

```
python main.py --modo completo --sintetico --amostras 2000 --modelo random_forest --estimadores 200 --max-depth 10
```

### Exemplo 2: Treinar com dados próprios e Regressão Logística

```
python main.py --modo treinar --dados data/meus_dados.csv --modelo logistic_regression
```

### Exemplo 3: Fazer predições com modelo existente

```
python main.py --modo prever --arquivo-modelo models/meu_modelo.pkl --dados-predicao data/dados_para_predicao.csv
```

## Módulos do Projeto

### preprocess.py

Contém funções para:
- Gerar dados sintéticos
- Carregar dados de arquivos CSV
- Limpar dados (remover valores ausentes e duplicatas)
- Normalizar dados
- Dividir dados em conjuntos de treino e teste

### train.py

Contém funções para:
- Criar modelos de classificação (Random Forest ou Regressão Logística)
- Treinar modelos
- Avaliar desempenho com métricas como acurácia e relatório de classificação
- Salvar modelos treinados

### predict.py

Contém funções para:
- Carregar modelos treinados
- Preparar novos dados para predição
- Realizar inferências
- Interpretar resultados

### main.py

Script principal que integra todos os módulos e fornece uma interface de linha de comando para:
- Executar o fluxo completo
- Apenas treinar um modelo
- Apenas fazer predições

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para mais detalhes.
