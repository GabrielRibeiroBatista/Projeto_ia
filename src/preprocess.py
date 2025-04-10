#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de pré-processamento de dados para o projeto de IA.

Este módulo contém funções para carregar ou gerar dados, realizar limpeza,
normalização e divisão dos dados em conjuntos de treino e teste.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def gerar_dados_sinteticos(
    n_amostras: int = 1000,
    n_features: int = 10,
    n_informative: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Gera um conjunto de dados sintéticos para classificação binária.

    Args:
        n_amostras: Número de amostras a serem geradas.
        n_features: Número total de características.
        n_informative: Número de características informativas.
        random_state: Semente para reprodutibilidade.

    Returns:
        DataFrame contendo os dados gerados com colunas nomeadas e rótulos.
    """
    print(f"Gerando {n_amostras} amostras de dados sintéticos com {n_features} características...")
    
    # Gerar dados sintéticos
    X, y = make_classification(
        n_samples=n_amostras,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_classes=2,
        random_state=random_state
    )
    
    # Criar nomes para as colunas
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    
    # Criar DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"Dados gerados com sucesso. Formato: {df.shape}")
    return df


def carregar_dados(caminho_arquivo: str) -> pd.DataFrame:
    """
    Carrega dados de um arquivo CSV.

    Args:
        caminho_arquivo: Caminho para o arquivo CSV.

    Returns:
        DataFrame contendo os dados carregados.
    
    Raises:
        FileNotFoundError: Se o arquivo não for encontrado.
    """
    print(f"Carregando dados do arquivo: {caminho_arquivo}")
    
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
    
    df = pd.read_csv(caminho_arquivo)
    print(f"Dados carregados com sucesso. Formato: {df.shape}")
    return df


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza limpeza básica nos dados.

    Args:
        df: DataFrame com os dados a serem limpos.

    Returns:
        DataFrame com os dados limpos.
    """
    print("Realizando limpeza dos dados...")
    
    # Verificar valores ausentes
    if df.isnull().sum().sum() > 0:
        print(f"Encontrados {df.isnull().sum().sum()} valores ausentes. Removendo linhas...")
        df = df.dropna()
    
    # Remover duplicatas
    duplicatas_antes = df.shape[0]
    df = df.drop_duplicates()
    duplicatas_removidas = duplicatas_antes - df.shape[0]
    
    if duplicatas_removidas > 0:
        print(f"Removidas {duplicatas_removidas} linhas duplicadas.")
    
    print(f"Limpeza concluída. Formato após limpeza: {df.shape}")
    return df


def normalizar_dados(
    X_train: np.ndarray,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Normaliza os dados de treino e teste usando StandardScaler.

    Args:
        X_train: Dados de treino.
        X_test: Dados de teste.

    Returns:
        Tupla contendo (X_train_normalizado, X_test_normalizado, scaler).
    """
    print("Normalizando dados...")
    
    # Criar e ajustar o scaler apenas nos dados de treino
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    
    # Aplicar a mesma transformação nos dados de teste
    X_test_norm = scaler.transform(X_test)
    
    print("Normalização concluída.")
    return X_train_norm, X_test_norm, scaler


def dividir_dados(
    df: pd.DataFrame,
    coluna_alvo: str = 'target',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide os dados em conjuntos de treino e teste.

    Args:
        df: DataFrame com os dados.
        coluna_alvo: Nome da coluna que contém os rótulos.
        test_size: Proporção do conjunto de teste.
        random_state: Semente para reprodutibilidade.

    Returns:
        Tupla contendo (X_train, X_test, y_train, y_test).
    """
    print(f"Dividindo dados em conjuntos de treino ({1-test_size:.0%}) e teste ({test_size:.0%})...")
    
    # Separar features e target
    X = df.drop(columns=[coluna_alvo]).values
    y = df[coluna_alvo].values
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Divisão concluída. X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def preparar_dados(
    caminho_arquivo: Optional[str] = None,
    usar_dados_sinteticos: bool = True,
    n_amostras: int = 1000,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list]:
    """
    Função principal que orquestra todo o processo de preparação dos dados.

    Args:
        caminho_arquivo: Caminho para o arquivo CSV (opcional).
        usar_dados_sinteticos: Se True, gera dados sintéticos em vez de carregar do arquivo.
        n_amostras: Número de amostras para dados sintéticos.
        test_size: Proporção do conjunto de teste.
        random_state: Semente para reprodutibilidade.

    Returns:
        Tupla contendo (X_train_norm, X_test_norm, y_train, y_test, scaler, feature_names).
    """
    # Carregar ou gerar dados
    if not usar_dados_sinteticos and caminho_arquivo:
        df = carregar_dados(caminho_arquivo)
    else:
        df = gerar_dados_sinteticos(n_amostras=n_amostras, random_state=random_state)
    
    # Armazenar nomes das features para uso posterior
    feature_names = df.columns.tolist()
    feature_names.remove('target')
    
    # Limpar dados
    df = limpar_dados(df)
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = dividir_dados(
        df, test_size=test_size, random_state=random_state
    )
    
    # Normalizar dados
    X_train_norm, X_test_norm, scaler = normalizar_dados(X_train, X_test)
    
    return X_train_norm, X_test_norm, y_train, y_test, scaler, feature_names


if __name__ == "__main__":
    # Testar o módulo quando executado diretamente
    X_train, X_test, y_train, y_test, scaler, feature_names = preparar_dados(
        usar_dados_sinteticos=True,
        n_amostras=500
    )
    
    print("\nResumo dos dados preparados:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    print(f"Número de features: {len(feature_names)}")
    print(f"Distribuição de classes (treino): {np.bincount(y_train)}")
    print(f"Distribuição de classes (teste): {np.bincount(y_test)}")
