## Configuração do Neon Database

1. Acessar o site https://neon.com/
2. Criar uma conta
3. Acessar `+ New project` (para este projeto foi usado o nome `financial-ml-decisioning`)
4. Criar um object storage (para este projeto foi usado o nome `model-artifacts`)
5. No repositório local, criar um .env (colocar no `.gitignore` para não subir para o repositório remoto)
6. No terminal, executar a sequência de comandos (compatíveis com MacOS):

    ``` bash
    # instalar a CLI do Neon (macOS)
    brew install neonctl

    # verificar a versão instalada do Neon CLI
    neonctl --version

    # fazer autenticação da CLI na conta do Neon
    neonctl auth

    # vincular o diretório/repositório local a um projeto existente no Neon
    neonctl link

    # inicializar a configuração do Neon no projeto e criar os arquivos de configuração necessários
    neon config init

    # instalar o pacote de configuração do Neon utilizado pelo arquivo neon.ts
    npm install @neon/config

    # verificar o plano de configuração antes de aplicar alterações no Neon
    neonctl config plan

    # aplicar no projeto Neon as alterações definidas na configuração local
    neonctl config deploy

    # obter as variáveis de ambiente necessárias para utilizar o Object Storage
    # e adicioná-las ao arquivo .env local
    neonctl env pull --service object-storage

    # listar somente os nomes das variáveis existentes no .env, sem exibir seus valores ou credenciais
    grep -o '^[A-Za-z_][A-Za-z0-9_]*=' .env

    # verificar se as variáveis do Object Storage foram carregadas no ambiente local
    # sem imprimir os valores secretos
    grep -E '^(AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_ENDPOINT_URL_S3|AWS_REGION|NEON_S3_BUCKET)=' .env | cut -d= -f1

    # nunca versionar o arquivo .env, pois ele contém credenciais e outros valores sensíveis
    echo ".env" >> .gitignore
    ```

7. Para fazer upload de tabelas: exemplo no notebook [5_upload_indicators_neon.ipynb](../notebooks/5_upload_indicators_neon.ipynb)