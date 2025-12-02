# RESUMO DO TRABALHO - Rede Neural em Prolog

## Enunciado no Google Classroom:
"Entrega de um vídeo descritivo e analítico (com aproximadamente 15 minutos de duração) com explicações e reflexões dos grupos de trabalho. Atenção: todos os integrantes dos grupos deverão participar das apresentações em vídeo.
Experimentação de Redes MLP (no Weka) para reconhecer letras (de A a Z) do nosso alfabeto.

Critérios de Avaliação:
- Elaboração e representação do dataset de todas as letras (A a Z) (3,0)
- Experimentações e reflexões acerca do processo de modelagem, aprendizagem e validação das redes neurais. (7,0)"

## Informações Gerais
- **Disciplina:** Inteligência Artificial (Faculdade)
- **Objetivo:** Criar rede neural em Prolog para reconhecer as 26 letras do alfabeto (A-Z)
- **Representação:** Matriz 5x5 (25 pixels), valores 1 e -1
- **Entregáveis:** Dataset + Código Prolog + Vídeo explicativo

---

## ✅ O QUE JÁ FOI FEITO

### 1. Dataset Criado
- **26 letras** do alfabeto desenhadas em matriz 5x5
- **6 variações por letra** (1 limpa + 5 com ruído de 1-3 pixels)
- **156 registros totais**
- Formatos: ARFF (WEKA) e fatos Prolog

### 2. Rede Neural Implementada
- **Arquitetura:** Perceptron multicamada (one-vs-all)
- **Entradas:** 25 neurônios (pixels)
- **Saídas:** 26 classificadores binários (um por letra)
- **Função de ativação:** Degrau (step function)
- **Fórmula:** `novo_peso = peso_atual + (taxa × erro × entrada)`

### 3. Funcionalidades do Código
- Menu interativo completo
- Inicialização de pesos aleatórios
- Treinamento por épocas configurável
- Avaliação com acurácia
- Avaliação detalhada (mostra erros)
- Teste visual de letras específicas
- Comparação automática de taxas de aprendizado
- Classificação de entrada manual

### 4. Documentação Criada
- Guia de instalação do SWI-Prolog
- Roteiro detalhado para gravação do vídeo
- Documentação visual das 26 letras

---

## 📁 ARQUIVOS GERADOS

| Arquivo | Descrição |
|---------|-----------|
| `rede_neural.pl` | **Código principal** - rede neural completa com dataset embutido |
| `letras_dataset.arff` | Dataset formato WEKA (156 exemplos com ruído) |
| `letras_dataset.pl` | Dataset formato Prolog separado |
| `letras_base.arff` | Dataset só com letras limpas (26 exemplos) |
| `letras_5x5.md` | Documentação visual de todas as 26 letras |
| `gerar_dataset.py` | Script Python para regenerar/ajustar dataset |
| `GUIA_USO.md` | Guia completo de instalação e uso |
| `ROTEIRO_VIDEO.md` | Roteiro detalhado para gravação do vídeo |

---

## 🔧 COMO USAR

### Instalação
1. Baixar SWI-Prolog: https://www.swi-prolog.org/download/stable
2. Instalar normalmente no Windows

### Execução
```prolog
% 1. Abrir SWI-Prolog

% 2. Carregar arquivo (usar / no caminho)
?- ['C:/SuaPasta/rede_neural.pl'].

% 3. Opção rápida - executa tudo
?- executar_tudo.

% 4. Ou usar menu interativo
?- menu.
```

### Comandos Principais
```prolog
?- inicializar_rede.      % Cria pesos aleatórios
?- treinar(100).          % Treina 100 épocas
?- avaliar.               % Mostra acurácia
?- avaliar_detalhado.     % Mostra erros específicos
?- testar_visual(a).      % Testa letra específica
?- comparar_taxas.        % Compara learning rates
```

---

## 🎥 ROTEIRO DO VÍDEO (10-15 min)

### Estrutura
1. **Introdução (2 min)** - Explicar o problema e objetivo
2. **Dataset (2 min)** - Mostrar estrutura das letras 5x5
3. **Arquitetura (2 min)** - Explicar Perceptron e fórmulas
4. **Demonstração (5 min)** - Executar no SWI-Prolog ao vivo
5. **Learning Rate (3 min)** - Mostrar `comparar_taxas` e analisar
6. **Conclusão (1 min)** - Resumir resultados

### Sequência de Comandos para Demonstração
```
1. Carregar arquivo
2. Opção 1: Inicializar rede
3. Opção 3: Avaliar (antes de treinar ~5%)
4. Opção 2: Treinar 10 épocas
5. Opção 3: Avaliar (~50%)
6. Opção 2: Treinar mais 90 épocas
7. Opção 4: Avaliação detalhada (~95-99%)
8. Opção 5: Testar letras a, m, z
9. Opção 9: Comparar taxas de aprendizado ← IMPORTANTE!
```

### Software para Gravar
- **Windows:** Win+G (Xbox Game Bar) ou OBS Studio
- **Dica:** Aumentar fonte do Prolog para ficar legível

---

## 📊 RESULTADOS ESPERADOS

### Acurácia por Fase
| Momento | Acurácia Esperada |
|---------|-------------------|
| Antes de treinar | ~5-10% |
| Após 10 épocas | ~40-60% |
| Após 100 épocas | ~95-99% |

### Comparação de Learning Rates (50 épocas)
| Taxa | Resultado |
|------|-----------|
| 0.01 | ~45% (muito lenta) |
| 0.05 | ~75% |
| 0.1 | ~90% |
| 0.2 | ~95% (ideal) |
| 0.3 | ~85% |
| 0.5 | ~70% (instável) |

### Erros Comuns
- B ↔ D (formato similar)
- O ↔ Q (diferem só na "cauda")
- Letras com poucos pixels característicos

---

## ⏳ PRÓXIMOS PASSOS

1. ✅ ~~Desenhar as 26 letras base~~
2. ✅ ~~Gerar variações com ruído~~
3. ✅ ~~Montar dataset (ARFF e Prolog)~~
4. ✅ ~~Implementar rede neural~~
5. ⏳ **Testar no SWI-Prolog local**
6. ⏳ **Ajustar parâmetros se necessário**
7. ⏳ **Gravar o vídeo**
8. ⏳ **Entregar trabalho**

---

## 💡 CONCEITOS PARA EXPLICAR NO VÍDEO

### Taxa de Aprendizado (Learning Rate)
- Controla o "tamanho do passo" no ajuste dos pesos
- Muito alta → oscila, não converge
- Muito baixa → aprende devagar demais
- Ideal para este problema: 0.1 a 0.2

### Época
- Uma passagem completa por todos os exemplos do dataset
- 100 épocas = ver todos os 156 exemplos 100 vezes

### One-vs-All
- Estratégia para classificação multiclasse
- 26 classificadores binários independentes
- Cada um aprende "é letra X?" vs "não é letra X?"

### Ruído no Dataset
- Simula imperfeições do mundo real
- Ajuda a rede generalizar melhor
- Evita overfitting (decorar os exemplos)

---

## 🔗 LINKS ÚTEIS

- **SWI-Prolog Download:** https://www.swi-prolog.org/download/stable
- **Documentação SWI-Prolog:** https://www.swi-prolog.org/pldoc/
- **OBS Studio (gravador):** https://obsproject.com/

---

*Última atualização: Novembro 2024*
