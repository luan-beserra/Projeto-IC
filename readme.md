## Projeto de Análise de Dados de Satélites e Detritos Espaciais

Este projeto foi desenvolvido ao longo do programa de Iniciação Científica na FATEC Diadema - Luigi Papaiz.

**Autor:** Luan Beserra de Oliveira.

**Orientadora:** Profª.  Me.  Andrea Zotovici.

**Este repositório é mantido como parte do meu portfólio pessoal.**

## Recursos Utilizados
**Python,  [pandas](https://pandas.pydata.org/), [PyCaret](https://github.com/pycaret/pycaret), [pyenv](https://github.com/pyenv/pyenv)**

## Desenvolvimento
De acordo com a problemática levantada, foi desenvolvido um algoritmo em **Python** com o objetivo de extrair dados de TLE entre satélites e detritos espaciais que estão em risco de colisão, segundo o site da [Space-Track](https://www.space-track.org/). Estes dados são filtrados, separados em um dataset e então processados de forma automatizada por vários modelos de aprendizagem de máquina de forma simultânea, utilizando conceitos de **autoML** com o auxílio da biblioteca [PyCaret](https://github.com/pycaret/pycaret). **Cada modelo faz uma previsão que calcula a probabilidade de colisão entre os dois objetos no período de 7 dias**, o algoritmo identifica qual o modelo mais preciso e mais eficiente, e permite que ele seja exportado para uso posterior. Também é gerado uma tabela que permite comparar os desempenhos de cada algotimo e comparar os resultados.

## Detalhes Teóricos da Proposta
### Título:
APLICAÇÃO DE ALGORITMOS DE APRENDIZAGEM DE MÁQUINA PARA A PREDIÇÃO DE DISTÂNCIAS MÍNIMAS ENTRE SATÉLITES E DETRITOS ESPACIAIS UTILIZANDO DADOS DE TLE

### Objetivos: 
Identificar as variáveis de entrada e os algoritmos de AM que oferecem o melhor desempenho na predição da menor distância entre um satélite e um detrito espacial utilizando dados de TLE.

### Justificativa: 
A implementação de métodos de AM em predições de distância de colisão é essencial para aumentar a precisão e eficiência de sistemas de monitoramento espacial, ajudando a prevenir possíveis colisões. A utilização de dados do TLE tem o potencial de permitir a predição com mais dias de antecedência permitindo maior tempo para planejar e executar a manobra do satélite. A escolha correta das variáveis de entrada e algoritmos pode não apenas melhorar a precisão dos modelos, mas também
reduzir o tempo de processamento necessário para gerar previsões. Além disso, o projeto atende a uma necessidade do setor aeroespacial e contribui para o desenvolvimento de técnicas que fortalecem a segurança espacial.

### Problemática:
Risco de colisões no espaço, que podem afetar diretamente operações de telecomunicação e monitoramento global.

## Configurações de Execução:
### pyenv:
**A ferramenta pyenv é usada para configurar o ambiente de execução**. No momento de desenvolvimento deste projeto, o PyCaret tem suporte apenas até a versão 11 do python, portanto é necessário usar esta ferramenta para **configurar a versão do python para 3.11.0b4** - qualquer versão inferior ao 11 deve ser compatível mas preferencialmente use a recomendada.

Após a instalação correta do pyenv, utilizando sua IDE de preferência, defina o interpretador python a ser usado para o pyenv. Caso utilizar o VS Code, siga o passo a passo:
    
     CTRL + SHIFT + P > Python: Select Interpreter > pyenv

### Instalação de dependências:
No terminal da IDE:

        pip install -r requirements.txt

### Variáveis de ambiente:
Um modelo base para configurar as variáveis de ambiente está presente no arquivo **.env.sample**.

A variável **IDENTITY** deve ser o e-mail de uma conta válida no site da [Space-Track](https://www.space-track.org/), **PASSWORD** é a senha vinculada ao e-mail.

**DATASET_PATH_CSV:** deve ser o caminho na sua máquina onde será salvo o arquivo.csv, que serve para exportar o dataset customizado que será criado pelo algoritmo. Recomendo que seja criada uma subpasta dentro do projeto, com o nome **datasets**, para que o arquivo seja facilmente encontrado.

**DATASET_PATH_XLS:** deve ser o caminho na sua máquina onde será salvo o arquivo.xlsx, que serve para exportar o dataset customizado que será criado pelo algoritmo. Recomendo que seja criada uma subpasta dentro do projeto, com o nome **datasets**, para que o arquivo seja facilmente encontrado.