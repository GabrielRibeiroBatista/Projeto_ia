#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Arquivo principal do projeto de IA de classificação.

Este arquivo integra os módulos de pré-processamento, treinamento e predição,
permitindo executar todo o fluxo de trabalho em uma única operação.
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# Importar módulos do projeto
from src.preprocess import preparar_dados
from src.train import treinar_e_avaliar
from src.predict import prever_novos_dados


def criar_parser() -> argparse.ArgumentParser:
    """
    Cria o parser de argumentos de linha de comando.

    Returns:
        Parser configurado.
    """
    parser = argparse.ArgumentParser(
        description='Sistema de IA para classificação binária',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Argumentos gerais
    parser.add_argument('--modo', type=str, default='completo',
                        choices=['completo', 'treinar', 'prever'],
                        help='Modo de operação: completo, treinar ou prever')
    
    # Argumentos para dados
    parser.add_argument('--dados', type=str, default=None,
                        help='Caminho para arquivo CSV com dados (opcional)')
    parser.add_argument('--sintetico', action='store_true',
                        help='Usar dados sintéticos em vez de arquivo')
    parser.add_argument('--amostras', type=int, default=1000,
                        help='Número de amostras para dados sintéticos')
    
    # Argumentos para treinamento
    parser.add_argument('--modelo', type=str, default='random_forest',
                        choices=['random_forest', 'logistic_regression'],
                        help='Tipo de modelo a ser usado')
    parser.add_argument('--estimadores', type=int, default=100,
                        help='Número de estimadores para Random Forest')
    parser.add_argument('--max-depth', type=int, default=None,
                        help='Profundidade máxima para Random Forest')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proporção do conjunto de teste')
    
    # Argumentos para predição
    parser.add_argument('--arquivo-modelo', type=str, 
                        default='models/modelo_treinado.pkl',
                        help='Caminho para o arquivo do modelo treinado')
    parser.add_argument('--dados-predicao', type=str, default=None,
                        help='Caminho para arquivo CSV com dados para predição')
    
    return parser


def executar_fluxo_completo(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Executa o fluxo completo: pré-processamento, treinamento e predição.

    Args:
        args: Argumentos de linha de comando.

    Returns:
        Dicionário com resultados do processo.
    """
    print("\n" + "="*50)
    print("INICIANDO FLUXO COMPLETO")
    print("="*50)
    
    # Definir caminho do modelo
    caminho_modelo = args.arquivo_modelo
    
    # Preparar dados
    print("\n[1/3] Preparando dados...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preparar_dados(
        caminho_arquivo=args.dados,
        usar_dados_sinteticos=args.sintetico or not args.dados,
        n_amostras=args.amostras,
        test_size=args.test_size
    )
    
    # Treinar e avaliar modelo
    print("\n[2/3] Treinando e avaliando modelo...")
    resultados_treino = treinar_e_avaliar(
        X_train, X_test, y_train, y_test, scaler, feature_names,
        tipo_modelo=args.modelo,
        caminho_modelo=caminho_modelo,
        n_estimators=args.estimadores,
        max_depth=args.max_depth
    )
    
    # Realizar predições em algumas amostras de teste
    print("\n[3/3] Testando predições em amostras do conjunto de teste...")
    # Usar 5 amostras do conjunto de teste para demonstração
    amostras_teste = X_test[:5]
    
    # Definir mapeamento de classes
    mapeamento_classes = {0: "Reprovado", 1: "Aprovado"}
    
    # Fazer predições
    resultados_predicao = prever_novos_dados(
        caminho_modelo,
        amostras_teste,
        mapeamento_classes=mapeamento_classes
    )
    
    # Mostrar resultados
    print("\nResultados da predição em amostras de teste:")
    print(resultados_predicao)
    
    # Comparar com valores reais
    y_real = y_test[:5]
    print("\nComparação com valores reais:")
    for i, (pred, real) in enumerate(zip(resultados_predicao['predicao'], y_real)):
        status = "✓" if pred == real else "✗"
        print(f"Amostra {i+1}: Predito={mapeamento_classes.get(pred)} | Real={mapeamento_classes.get(real)} {status}")
    
    print("\n" + "="*50)
    print(f"FLUXO COMPLETO FINALIZADO COM SUCESSO!")
    print(f"Modelo salvo em: {resultados_treino['caminho_modelo']}")
    print(f"Acurácia: {resultados_treino['acuracia']:.4f}")
    print("="*50)
    
    return {
        'resultados_treino': resultados_treino,
        'resultados_predicao': resultados_predicao
    }


def executar_apenas_treinamento(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Executa apenas o fluxo de pré-processamento e treinamento.

    Args:
        args: Argumentos de linha de comando.

    Returns:
        Dicionário com resultados do treinamento.
    """
    print("\n" + "="*50)
    print("INICIANDO TREINAMENTO")
    print("="*50)
    
    # Definir caminho do modelo
    caminho_modelo = args.arquivo_modelo
    
    # Preparar dados
    print("\n[1/2] Preparando dados...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preparar_dados(
        caminho_arquivo=args.dados,
        usar_dados_sinteticos=args.sintetico or not args.dados,
        n_amostras=args.amostras,
        test_size=args.test_size
    )
    
    # Treinar e avaliar modelo
    print("\n[2/2] Treinando e avaliando modelo...")
    resultados_treino = treinar_e_avaliar(
        X_train, X_test, y_train, y_test, scaler, feature_names,
        tipo_modelo=args.modelo,
        caminho_modelo=caminho_modelo,
        n_estimators=args.estimadores,
        max_depth=args.max_depth
    )
    
    print("\n" + "="*50)
    print(f"TREINAMENTO FINALIZADO COM SUCESSO!")
    print(f"Modelo salvo em: {resultados_treino['caminho_modelo']}")
    print(f"Acurácia: {resultados_treino['acuracia']:.4f}")
    print("="*50)
    
    return resultados_treino


def executar_apenas_predicao(args: argparse.Namespace) -> pd.DataFrame:
    """
    Executa apenas o fluxo de predição com um modelo já treinado.

    Args:
        args: Argumentos de linha de comando.

    Returns:
        DataFrame com os resultados das predições.
        
    Raises:
        ValueError: Se não for fornecido um arquivo de dados para predição.
    """
    print("\n" + "="*50)
    print("INICIANDO PREDIÇÃO")
    print("="*50)
    
    # Verificar se foi fornecido arquivo de dados para predição
    if not args.dados_predicao and not args.sintetico:
        raise ValueError("É necessário fornecer um arquivo de dados para predição ou usar a opção --sintetico")
    
    # Carregar dados para predição
    if args.dados_predicao:
        print(f"\n[1/2] Carregando dados de {args.dados_predicao}...")
        dados_predicao = pd.read_csv(args.dados_predicao)
    else:
        print("\n[1/2] Gerando dados sintéticos para predição...")
        from src.preprocess import gerar_dados_sinteticos
        dados_temp = gerar_dados_sinteticos(n_amostras=10)
        dados_predicao = dados_temp.drop(columns=['target'])
    
    # Definir mapeamento de classes
    mapeamento_classes = {0: "Reprovado", 1: "Aprovado"}
    
    # Fazer predições
    print(f"\n[2/2] Realizando predições com o modelo {args.arquivo_modelo}...")
    resultados = prever_novos_dados(
        args.arquivo_modelo,
        dados_predicao,
        mapeamento_classes=mapeamento_classes
    )
    
    # Mostrar resultados
    print("\nResultados da predição:")
    print(resultados)
    
    print("\n" + "="*50)
    print("PREDIÇÃO FINALIZADA COM SUCESSO!")
    print("="*50)
    
    return resultados


def main():
    """
    Função principal que coordena a execução do programa.
    """
    # Criar parser e obter argumentos
    parser = criar_parser()
    args = parser.parse_args()
    
    # Executar modo selecionado
    try:
        if args.modo == 'completo':
            resultados = executar_fluxo_completo(args)
        elif args.modo == 'treinar':
            resultados = executar_apenas_treinamento(args)
        elif args.modo == 'prever':
            resultados = executar_apenas_predicao(args)
        else:
            raise ValueError(f"Modo inválido: {args.modo}")
    
    except Exception as e:
        print(f"\nERRO: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
