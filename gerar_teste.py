#!/usr/bin/env python3
"""
Gerador de Dataset de TESTE para Validacao da Rede Neural
Gera letras com ruido DIFERENTE do dataset de treino (4-5 pixels)
"""

import random
from typing import List

# Letras base (mesmo padrao do dataset original)
LETRAS_BASE = {
    'A': [-1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1],
    'B': [1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1],
    'C': [-1,1,1,1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,1,1,1],
    'D': [1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,-1],
    'E': [1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1],
    'F': [1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1],
    'G': [-1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1],
    'H': [1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1],
    'I': [1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,1],
    'J': [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1],
    'K': [1,-1,-1,-1,1,1,-1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,1,-1,1,-1,-1,-1,1],
    'L': [1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,1],
    'M': [1,-1,-1,-1,1,1,1,-1,1,1,1,-1,1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1],
    'N': [1,-1,-1,-1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,-1,-1,1,1,1,-1,-1,-1,1],
    'O': [-1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,1,1,-1],
    'P': [1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1],
    'Q': [-1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,-1,-1,1,1,-1,1],
    'R': [1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,-1,1,-1,1,-1,-1,-1,1],
    'S': [-1,1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,-1],
    'T': [1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1],
    'U': [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,1,1,-1],
    'V': [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1],
    'W': [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,1,1,1,-1,1,1,1,-1,-1,-1,1],
    'X': [1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1],
    'Y': [1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1],
    'Z': [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,1,1],
}


def adicionar_ruido(padrao: List[int], num_pixels: int) -> List[int]:
    """Adiciona ruido invertendo pixels aleatorios."""
    novo_padrao = padrao.copy()
    indices = random.sample(range(25), num_pixels)
    for i in indices:
        novo_padrao[i] = -novo_padrao[i]
    return novo_padrao


def visualizar_letra(padrao: List[int], titulo: str = ""):
    """Exibe visualmente uma letra no console."""
    if titulo:
        print(f"\n{titulo}")
    print("-" * 11)
    for i in range(5):
        linha = ""
        for j in range(5):
            idx = i * 5 + j
            linha += "# " if padrao[idx] == 1 else ". "
        print(f"| {linha}|")
    print("-" * 11)


def gerar_teste_arff(filename: str, num_exemplos_por_letra: int = 2, ruido_min: int = 4, ruido_max: int = 5):
    """
    Gera arquivo ARFF de teste com ruido diferente do treino.
    
    Args:
        filename: Nome do arquivo de saida
        num_exemplos_por_letra: Quantos exemplos por letra
        ruido_min: Minimo de pixels de ruido
        ruido_max: Maximo de pixels de ruido
    """
    
    with open(filename, 'w') as f:
        # Header ARFF
        f.write("@relation letras_alfabeto_teste\n\n")
        
        for i in range(1, 26):
            f.write(f"@attribute n{i} {{-1, 1}}\n")
        
        f.write("@attribute classe {A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z}\n\n")
        f.write("@data\n")
        
        # Gerar dados de teste
        total_registros = 0
        for letra, padrao in LETRAS_BASE.items():
            for _ in range(num_exemplos_por_letra):
                # Ruido entre 4 e 5 pixels (diferente do treino que usa 1-3)
                nivel_ruido = random.randint(ruido_min, ruido_max)
                padrao_ruido = adicionar_ruido(padrao, nivel_ruido)
                linha = ','.join(str(v) for v in padrao_ruido) + f",{letra}\n"
                f.write(linha)
                total_registros += 1
        
        print(f"Arquivo {filename} gerado com {total_registros} registros!")
        print(f"Ruido aplicado: {ruido_min}-{ruido_max} pixels (mais que o treino)")


def demonstrar_diferenca():
    """Mostra visualmente a diferenca entre treino e teste."""
    
    print("\n" + "=" * 50)
    print("COMPARACAO: TREINO vs TESTE")
    print("=" * 50)
    
    letra = 'A'
    padrao = LETRAS_BASE[letra]
    
    visualizar_letra(padrao, f"Letra {letra} - ORIGINAL")
    
    # Ruido de treino (1-3 pixels)
    padrao_treino = adicionar_ruido(padrao, 2)
    visualizar_letra(padrao_treino, f"Letra {letra} - TREINO (2 pixels ruido)")
    
    # Ruido de teste (4-5 pixels)
    padrao_teste = adicionar_ruido(padrao, 5)
    visualizar_letra(padrao_teste, f"Letra {letra} - TESTE (5 pixels ruido)")
    
    print("\nObserve: o teste tem MAIS ruido que o treino!")
    print("Isso forca a rede a generalizar, nao decorar.")


if __name__ == "__main__":
    # Seed para reprodutibilidade (mude para gerar variacoes diferentes)
    random.seed(123)
    
    print("=" * 60)
    print("GERADOR DE DATASET DE TESTE")
    print("=" * 60)
    
    # Gerar arquivo de teste
    # 2 exemplos por letra = 52 registros de teste
    gerar_teste_arff("letras_teste.arff", num_exemplos_por_letra=2, ruido_min=4, ruido_max=5)
    
    # Demonstrar diferenca visual
    demonstrar_diferenca()
    
    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    print("Dataset de TREINO: 156 exemplos, ruido de 1-3 pixels")
    print("Dataset de TESTE:  52 exemplos, ruido de 4-5 pixels")
    print("\nO teste usa ruido MAIOR para verificar generalizacao!")
