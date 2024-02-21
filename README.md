# Reconhecimento Facial - Projeto

## Integrantes

- Anabel Marinho Soares
- Nicolas Emanuel Alves Costa
- Thiago Luan Moreira Sousa

Este projeto visa desenvolver um programa capaz de utilizar uma rede neural treinada para detectar expressões faciais nos rostos dos usuários por meio de suas câmeras.

Atualmente, existem duas versões disponíveis do sistema, diferenciadas pelo modelo de rede neural utilizada:

1. **Modelo MLP:**
   - `treinamentoMLP.py`
   - `cameraMLP.py`
   - `versaoMLP.ipynb`

2. **Modelo CNN:**
   - `treinamentoCNN.py`
   - `cameraCNN.py`
   - `versaoCNN.py`

Link para download do dataset utilizado: [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

Para executar o programa, é necessário instalar as dependências listadas no arquivo "requirements.txt" e iniciar um ambiente virtual, como o "venv".

Após isso, basta iniciar o Streamlit chamando o arquivo do aplicativo desejado no terminal. Por exemplo: `streamlit run cameraCNN.py`.