# Projeto1
Dinho

Main.py — Pipeline de medição de mudas de eucalipto.

Caso rode com o código sem partes comentadas e para as 5 primeiras imagens:
 
Para cada imagem do conjunto:
  1. Carrega e padroniza a altura para 2000 px (fator de escala salvo).
  2. Segmenta planta + vaso por cor de fundo.
  3. Detecta o topo e centro do vaso (referência para base do caule).
  4. Gera duas máscaras da planta (suave para o caule, rude para a altura).
  5. Esqueletiza a máscara suave.
  6. Identifica base e topo do caule no skeleton.
  7. Traça o caule com Dijkstra ponderado, poda em 0.85·h e estende
     pelo skeleton se houver caule fino acima.
  8. Mede:
        - altura vertical (a partir da máscara rude);
        - comprimento do caule (em pixels do caminho);
        - diâmetro do coleto (10 unidades acima do vaso, na imagem original).
  9. Converte tudo de volta para a escala original (dividindo por F_escala).
 10. Compara com o gabarito e acumula MAPEs
 
No fim imprime a tabela com os resultados e os MAPEs médios.

Caso o código seja rodado no estado atual:

Para cada imagem do conjunto:
  1. Carrega e padroniza a altura para 2000 px (fator de escala salvo).
  2. Segmenta planta + vaso por cor de fundo.
  3. Detecta o topo e centro do vaso (referência para base do caule).
  4. Gera duas máscaras da planta (suave para o caule, rude para a altura).
  5. Esqueletiza a máscara suave.
  6. Identifica base e topo do caule no skeleton.
  7. Traça o caule com Dijkstra ponderado, poda em 0.85·h e estende
     pelo skeleton se houver caule fino acima.
  8. Mede:
        - altura vertical (a partir da máscara rude);
        - comprimento do caule (em pixels do caminho);
        - diâmetro do coleto (10 unidades acima do vaso, na imagem original).
  9. Converte tudo de volta para a escala original (dividindo por F_escala).
 
No fim imprime a tabela com os resultados
