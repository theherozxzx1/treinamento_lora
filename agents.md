Perfeito. Vou buscar a melhor forma de configurar e realizar um treinamento de LoRA de identidade visual facial com 18 imagens em resolução 1024x1024, utilizando o modelo FLUX DEV no Google Colab com GPU A100. Também irei incluir as etapas de validação para avaliar a qualidade do LoRA treinado a partir dos prompts fornecidos.

Já retorno com um guia completo para você.


# Guia de Treinamento LoRA com o modelo FLUX.1-Dev no Google Colab (GPU A100)

## 1. Configuração do ambiente no Google Colab (GPU A100)

Para treinar um LoRA usando o modelo **FLUX.1-Dev** no Google Colab, é necessário preparar o ambiente com GPU de alto desempenho (A100) e instalar as dependências adequadas. Siga os passos abaixo:

1. **Ative a GPU A100 no Colab:** No menu do Colab, vá em *Runtime > Change runtime type* (Alterar tipo de runtime) e selecione **GPU**. Para garantir acesso a uma GPU A100 e mais memória, use o Colab Pro/Pro+ e habilite a opção de **High-RAM** (mais memória). *Observação:* O treinamento de Flux LoRA em Colab **exige** uma conta paga (Pro/Pro+) devido ao alto consumo de recursos. Com Colab Pro, você normalmente obtém GPUs L4 ou A100 com \~40 GB VRAM, o que é suficiente para o Flux.

2. **Clone os repositórios do FluxGym e Kohya:** Abra uma célula no Colab e execute os comandos para baixar o código do FluxGym (uma interface para treinar LoRAs do Flux) e o script de treinamento LoRA (Kohya ss/sd-scripts):

   ```bash
   !git clone https://github.com/TheLocalLab/fluxgym-Colab.git 
   %cd /content/fluxgym-Colab/
   !git clone -b sd3 https://github.com/kohya-ss/sd-scripts
   ```

   Os comandos acima criam a pasta `fluxgym-Colab` com o FluxGym e baixam o repositório `sd-scripts` (branch sd3) dentro dela, contendo os scripts de treinamento LoRA do Kohya.

3. **Instale as dependências necessárias:** Em seguida, instale os pacotes Python requeridos tanto pelo sd-scripts quanto pelo FluxGym:

   ```bash
   %cd /content/fluxgym-Colab/sd-scripts
   !pip install -r requirements.txt
   %cd /content/fluxgym-Colab/
   !pip install -r requirements.txt
   # Instalar versão do PyTorch compatível (Nightly cu121 para A100, com suporte a FP8)
   !pip install --pre torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   Esses comandos vão instalar bibliotecas como `accelerate`, `transformers`, `safetensors`, etc., necessárias para o treinamento. Note que usamos uma versão **Nightly do PyTorch 2.4** com CUDA 12.1 (`--pre torch==2.4 ... cu121`) porque o Flux.1-Dev utiliza pesos em formato FP8 e precisamos de suporte adequado a esse tipo numérico.

4. **Baixe os arquivos do modelo FLUX.1-Dev e componentes:** O modelo Flux é composto por múltiplos arquivos (devido ao seu tamanho e arquitetura). Crie as pastas correspondentes dentro de `fluxgym-Colab/models/` e baixe os seguintes arquivos do HuggingFace:

   ```bash
   # Crie as pastas se necessário
   !mkdir -p /content/fluxgym-Colab/models/unet
   !mkdir -p /content/fluxgym-Colab/models/clip
   !mkdir -p /content/fluxgym-Colab/models/vae
   # 4.a) Modelo UNet do Flux (peso principal do difusor, FP8)
   !wget -O /content/fluxgym-Colab/models/unet/flux1-dev-fp8.safetensors \
        https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors
   # 4.b) Text Encoder CLIP_L (modelo de texto complementar do Flux)
   !wget -O /content/fluxgym-Colab/models/clip/clip_l.safetensors \
        https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true
   # 4.c) Text Encoder T5-XXL (segundo encoder de texto, quantizado em FP8)
   !wget -O /content/fluxgym-Colab/models/clip/t5xxl_fp8.safetensors \
        https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors?download=true
   # 4.d) Autoencoder Variacional (VAE) do Flux
   !wget -O /content/fluxgym-Colab/models/vae/ae.sft \
        https://huggingface.co/cocktailpeanut/xulf-dev/resolve/main/ae.sft?download=true
   ```

   **Descrição dos arquivos baixados:**

   * `flux1-dev-fp8.safetensors` – modelo **UNet** do Flux Dev (peso principal do modelo de difusão, formato FP8). Essa é a parte responsável por gerar imagens a partir do ruído.
   * `clip_l.safetensors` – modelo **CLIP-L** (Large) que serve como **codificador de texto** adicional para o Flux. O Flux usa dois encoders de texto: este (derivado de CLIP, captura embeddings de texto/imagem) e um T5.
   * `t5xxl_fp8.safetensors` – modelo **T5-XXL** (quantizado em FP8) usado como **segundo codificador de texto** do Flux. Ele interpreta o prompt em linguagem natural, complementando o CLIP.
   * `ae.sft` – modelo **VAE** (Autoencoder Variacional) do Flux, responsável por codificar/decodificar as imagens (compressão dos outputs em latentes e reconstrução da imagem final).

   *Observações:* O modelo **Flux.1-Dev** é disponibilizado sob licença não-comercial e requer aceitação dos termos na HuggingFace para acesso. Nos comandos acima, utilizamos repositórios alternativos (como `Kijai/flux-fp8` e `comfyanonymous/flux_text_encoders`) que fornecem os arquivos diretamente, possivelmente sem exigir autenticação. Se esses links falharem, você deve:<br>
   a) Fazer login na HuggingFace (`!huggingface-cli login`) com uma conta que tenha aceitado os termos do *FLUX.1-dev*, ou<br>
   b) Baixar os arquivos manualmente e fazer upload para o Colab ou seu Google Drive. <br>
   Certifique-se de ter todos os quatro componentes (UNet, CLIP, T5, VAE) correspondentes à versão do Flux Dev que pretende usar. Sem eles, o modelo não funcionará corretamente.

5. **Execute o servidor FluxGym:** Com tudo instalado, rode o aplicativo FluxGym. Na célula do Colab, execute:

   ```bash
   !python app.py
   ```

   Isso iniciará a interface web (Gradio) do FluxGym no Colab. Aguarde a inicialização; ao terminar, o log exibirá um **Public URL** (via *gradio* ou *ngrok*) onde você poderá acessar a interface gráfica do FluxGym usando o navegador do seu computador. Clique nesse link ou copie e abra em uma nova aba: ele redireciona para a UI do FluxGym rodando no ambiente Colab.

   *Dica:* Caso o Colab use ngrok para tunelamento, pode ser solicitada a criação de um token de autenticação ngrok. Siga as instruções do log (forneça o token conforme indicado, se necessário, para gerar o link público).

   A interface do FluxGym é dividida em três partes principais:

   * **LoRA Info:** campos para definir nome do modelo LoRA, trigger words, configuração de VRAM e hiperparâmetros de treino.

   * **Dataset:** seção para fazer upload das imagens de treino e fornecer legendas (captions) para cada uma.

   * **Training:** mostra os comandos/configurações finalizados e um botão para iniciar o treinamento, além de opcionalmente exibir logs ou amostras durante o processo.

   > **Nota:** Mantenha a aba do Colab aberta e em atividade durante todo o treinamento para evitar que o notebook seja finalizado por inatividade. Usuários Colab Pro têm menor chance de desconexão, mas ainda é recomendável monitorar o progresso. Você pode, por exemplo, deixar uma célula fazendo pequenos outputs periódicos ou utilizar a extensão *Colab Alive* como medida preventiva.

## 2. Preparação do dataset com as 18 imagens (1024x1024)

Antes de iniciar o treinamento, é fundamental preparar adequadamente o conjunto de imagens do seu dataset. No caso, você possui 18 imagens de um rosto/identidade em resolução 1024x1024. Seguem as boas práticas de preparação:

* **Resolução e formato das imagens:** Garanta que as imagens estejam em **formato PNG** (evite JPG/JPEG, que podem causar erros no pipeline do Flux) e com resolução próxima de **1024×1024** pixels. O Flux foi projetado para trabalhar com imagens quadradas (*aspect ratio* 1:1), então se alguma foto não for quadrada, corte ou redimensione para 1024x1024 (ou pelo menos proporção 1:1) com o sujeito centralizado. Usar imagens já na resolução de treino ajuda a manter qualidade e evita surpresas no recorte automático. (Tamanhos diferentes também funcionam, mas inclua algumas imagens exatamente em 1024×1024 para melhor desempenho do modelo).

* **Quantidade de imagens:** 18 imagens é um bom tamanho de dataset para identidade facial. Em geral, recomenda-se **10–20 imagens** de alta qualidade para treinar o rosto de uma pessoa. Quantidades maiores podem melhorar a diversidade, mas também aumentam o tempo de treinamento e o risco de ruído; quantidades menores que \~10 imagens podem resultar em um modelo menos flexível ou que sobre-ajusta demais cada exemplo. Com 18 fotos, você está dentro da faixa ideal para capturar a identidade sem exagerar nos detalhes específicos de uma única imagem.

* **Diversidade e cobertura:** A **diversidade é chave** para um bom treinamento. Use fotos do sujeito em diferentes **situações, cenários, ângulos de câmera, expressões faciais, poses e iluminações**. Por exemplo, inclua algumas fotos sorrindo, outras sério; algumas em ambiente interno, outras externo; diferentes roupas e acessórios, etc. Essa variação ensina o modelo a generalizar a identidade da pessoa, em vez de **memorizar uma pose ou fundo específico**. Se todas as 18 imagens fossem muito parecidas (mesma roupa ou cenário), o LoRA poderia confundir esses atributos com a identidade e reproduzi-los em toda geração. Portanto, tente cobrir o **maior espectro visual** possível do seu sujeito dentro do limite de imagens.

* **Qualidade das imagens:** Use apenas **imagens nítidas e de boa qualidade**. Rostos devem estar **focados e centralizados**, sem borrões ou resolução baixa. Remova fotos com artefatos, marcas d'água ou muito ruído. Também evite imagens em que o rosto esteja muito pequeno no quadro – close-ups ou meio-corpo funcionam melhor para capturar detalhes faciais. Incluir **alguns headshots (close do rosto)** em alta resolução é recomendável para ensinar detalhes finos dos traços. Cada imagem deve *complementar* as demais, adicionando informação nova (ex: um ângulo diferente ou expressão distinta), ao invés de repetir praticamente a mesma cena.

* **Consistência do sujeito:** O dataset deve conter **apenas o sujeito de interesse** como figura central. Evite fotos com múltiplas pessoas se possível. Caso haja outras pessoas ou personagens presentes, o modelo pode se confundir sobre quem é o alvo do LoRA. Se não tiver como remover, **mencione na legenda** quantas pessoas há ou quem é quem, para que o treinamento não atribua erroneamente características de outra pessoa ao seu sujeito. Por exemplo: se uma imagem tem o sujeito posando ao lado de um amigo, a legenda pode dizer "PessoaXYZ ao lado de um amigo" para o modelo saber que nem tudo naquela imagem deve ser aprendido como parte da identidade "PessoaXYZ". O mesmo vale para objetos muito destacados ou pets junto do sujeito – clarifique na legenda ou prefira imagens solo.

* **Legendas (captions) e *trigger word*:** Para cada imagem, prepare uma **legenda de treino** que descreva brevemente a cena **incluindo uma palavra-chave exclusiva para o sujeito**, chamada aqui de *trigger word*. Exemplo: se o nome da pessoa é Ana, você pode usar um codinome único como `"anafox"` (algo pouco comum para evitar conflitos com conhecimento pré-existente do modelo) e inseri-lo nas legendas. No caso de **treino de um único rosto humano**, Flux permite até treinar sem legendas detalhadas, mas **recomenda-se fortemente usar pelo menos o trigger** em cada imagem para ligar o conceito àquela pessoa. Você pode manter as legendas simples, apenas dizendo `"foto de anafox"` em cada uma, ou acrescentar alguns detalhes relevantes da imagem (por exemplo: `"anafox sorrindo, usando óculos escuros"`, `"anafox em pé no parque, ao pôr-do-sol"`, etc.). O importante é que *todas* as legendas contenham a mesma palavra única (trigger) para representar o seu sujeito. Isso fará com que, durante o treinamento, o LoRA aprenda a associar essa **palavra** às características visuais da pessoa nas fotos, em vez de tentar aprender “anonimamente” cada imagem. Como resultado, quando você quiser gerar imagens, bastará usar essa palavra no prompt para invocar a identidade. Além disso, legendas ajudam a guiar o modelo caso alguma imagem tenha elementos incomuns – por exemplo, se o sujeito está de chapéu em uma foto, mencionar "usando chapéu" na legenda pode evitar que o modelo pense que o chapéu faz parte inerente da identidade (pode parecer contraintuitivo, mas adicionar contexto real ajuda a não sobrevalorizar aquele detalhe específico).

  *Dicas para legendas:* Se preferir, use alguma ferramenta de auto-caption para ganhar tempo – o FluxGym tem suporte ao modelo **Florence** para legendar imagens automaticamente com um clique. Ele vai gerar descrições das cenas; você então edita cada legenda inserindo o **trigger word** nelas (por exemplo, trocar "a woman with blonde hair smiling" por "anafox, blonde woman smiling"). Lembre-se de manter as legendas **consistentes**: utilize sempre o mesmo nome-chave escrito de forma idêntica (respeitando maiúsculas/minúsculas). Pequenas variações podem ser interpretadas como coisas diferentes pelo modelo. Se estiver em dúvida sobre o nível de detalhe das captions: para humanos reais, muitos treinamentos bem-sucedidos usam **somente o nome/trigger** sem mais nada, confiando que o modelo base já entende características humanas gerais. Já para estilos artísticos ou personagens de ficção, costuma-se precisar de legendas mais detalhadas. No seu caso (identidade visual facial), começar com legendas curtas + trigger deve funcionar, mas sinta-se à vontade para enriquecer a descrição se notar problemas.

Resumindo: tenha suas 18 imagens **cortadas para 1:1**, em **PNG**, com boa qualidade e **variadas**. Crie uma legenda para cada uma contendo a **mesma palavra-chave exclusiva** do sujeito (e eventualmente descrições básicas). Coloque todas essas imagens em uma pasta ou zip para fácil acesso no Colab. Você pode enviá-las para o Colab de diferentes formas: via Google Drive (montando o drive no notebook), via upload direto pela interface FluxGym (há um botão para isso), ou usando `files.upload()` no Colab para enviar arquivos locais. A seguir, partiremos para o treinamento usando esses dados.

## 3. Treinamento do LoRA no FLUX.1-Dev

Com o ambiente pronto e os dados organizados, podemos iniciar o treinamento do LoRA no modelo Flux.1-Dev. Iremos utilizar a interface gráfica do **FluxGym** que abrimos anteriormente (passo 1.5) para configurar e conduzir o treino. Abaixo, detalhamos cada etapa do processo e os parâmetros recomendados para o dataset de 18 imagens em 1024²:

1. **Acesse a interface FluxGym:** Abra o link público do Gradio (fornecido no log do Colab após rodar `app.py`) em seu navegador. Você verá a página do FluxGym, geralmente com três colunas ou seções (LoRA Info, Dataset, Training).

2. **Preencha as informações do LoRA (Seção *LoRA Info*):** Nesta parte, você define os metadados e hiperparâmetros do treinamento:

   * **LoRA Name (Nome do modelo):** escolha um nome descritivo para seu LoRA, por exemplo, `fluxLora_Ana` ou algo que identifique o sujeito. Esse será o nome do arquivo `.safetensors` gerado ao final (não use espaços ou caracteres especiais, para segurança).

   * **Trigger Word:** insira exatamente a palavra-chave que você usou nas legendas (por exemplo, `anafox`). É crucial que seja idêntica ao utilizado nas captions, incluindo maiúsculas/minúsculas.

   * **VRAM/GPU Setting:** selecione na interface a configuração de memória da GPU. O FluxGym costuma ter predefinições como 12GB, 16GB, 20GB etc. Para uma A100 40GB, escolha a opção mais alta disponível (por exemplo, *20GB* ou *24GB* se houver). Isso permitirá usar batch size maior e/ou modelos em maior precisão. *Nota:* Versões antigas do FluxGym indicavam melhor estabilidade na opção 16GB em vez de 20GB, mas sinta-se livre para usar 20GB se não houver problemas – com 40GB físicos você não deve enfrentar Out Of Memory.

   * **Repeat count (Repeat trains per image):** defina o número de repetições por imagem por época. Esse parâmetro, junto com o número de épocas e quantidade de imagens, determina o total de passos de treino. Uma **regra geral** proveniente de experiências com Stable Diffusion/SDXL é **\~100 passos por imagem** do dataset para um bom resultado. Podemos adotar a mesma lógica aqui: com 18 imagens, 100 passos/imagem resultariam em \~1800 passos no total. Você pode conseguir isso, por exemplo, definindo **Repeat = 10**.

   * **Max Train Epochs:** defina o número de épocas (passes completos pelo dataset). Seguindo o exemplo, se Repeat = 10 e temos 18 imagens, cada época terá 18\*10 = 180 *passos*. Para chegar perto de 1800 passos, precisaríamos de 10 épocas (pois 180 \* 10 = 1800). Portanto, coloque **Epochs = 10**. Assim, o treinamento percorrerá cada imagem 10 vezes por época, por 10 épocas. O próprio FluxGym ou sd-scripts deve mostrar o *Expected training steps* calculado (por ex., 1800) para confirmar. Caso queira um ajuste fino, você pode diminuir/incrementar ligeiramente esse valor total. Mas evite exceder muito 100–150 passos/imagem inicialmente, para não arriscar overfitting.

   * **Outros hiperparâmetros básicos:** muitos itens podem vir com valores padrão adequados. Por exemplo, *Learning rate* (taxa de aprendizado) costuma defaultar para **`1e-4`**, o que é um bom ponto de partida na maioria dos casos. Verifique se está em torno de 1e-4 a 2e-4; não há necessidade de começar com algo muito diferente a menos que tenha experiência específica. *Unet learning rate* e *Text Encoder learning rate* às vezes são separados – se estiverem, deixe ambos iguais (1e-4). Parâmetros como *warmup steps*, *optimizer* etc. podem ficar nos defaults recomendados pelo FluxGym/Kohya.

   * **Network Dimension/Rank (dimensões do LoRA):** se a interface expuser essa opção (pode estar na aba avançada), escolha o *rank* do LoRA. Esse valor controla a capacidade do LoRA em termos de graus de liberdade (é o número de dimensões latentes inseridas em cada camada treinável). O padrão em alguns scripts pode ser baixo (por exemplo, 4 ou até 2), mas isso pode limitar a expressividade do LoRA. Recomenda-se usar algo em torno de **16 ou 32** para obter melhor qualidade e fidelidade. Ranks maiores aumentam o tamanho do arquivo final (ex.: rank 16 resulta num LoRA de \~30 MB; rank 32 o dobro disso, aproximadamente) mas capturam mais detalhes. Com 18 imagens, rank 16 geralmente basta; se quiser máxima qualidade e sua VRAM permitir, 32 é o teto razoável. Evite ranks muito altos (64+) pois podem sobreparametrizar e acabar memorizando demais o dataset pequeno. Por outro lado, ranks muito baixos (por ex. 2 ou 4) podem fazer o LoRA incapaz de reproduzir características importantes do rosto.

   * **Batch size:** se houver opção para *batch size* (tamanho do lote), ajuste conforme sua VRAM. Treinar com batch > 1 faz o modelo processar múltiplas imagens em cada passo, o que *efetivamente aumenta o contexto e pode levar a resultados melhores ou treinamento mais estável*. Com uma A100 40GB, você deve conseguir batch 2 ou 4 tranquilamente em 1024x1024. Por exemplo, se puder, use **batch\_size = 2** (isso duplicará o número de imagens vistas por passo, reduzindo pela metade o tempo necessário para mesma quantidade de epochs, ou permitindo mais passos no mesmo tempo). Só tome cuidado: se o batch for muito grande, existe chance de estourar a memória ou diminuir a efetividade da descida de gradiente por ruído reduzido. Batch 2 a 4 é um bom compromisso se possível.

   > *Resumo:* Para 18 imagens 1024², configuração sugerida: **Repeat 10, Epochs 10 (≈1800 passos)**, **Learning rate \~1e-4**, **Rank 16**, **Batch 2** (se possível). Esses valores se baseiam em práticas recomendadas e devem capturar bem a identidade sem overfitting severo, dado um dataset diversificado.

3. **Carregue as imagens e aplique as legendas (Seção *Dataset*):** Após definir os hiperparâmetros, vá para a seção de Dataset no FluxGym. Ali haverá a opção de fazer **upload das imagens** de treinamento. Você pode arrastar e soltar todas as 18 imagens de uma vez (ou selecioná-las via diálogo de arquivo). Elas serão listadas na interface. Para cada imagem enviada, haverá um campo de texto para inserir a **legenda (caption)** correspondente. Preencha cada um **exatamente** com a legenda preparada (incluindo o trigger word). Certifique-se de que não esqueceu nenhuma imagem sem legenda ou vice-versa. Se você já montou o Drive com as imagens ou as copiou para alguma pasta em `/content`, o FluxGym pode ter uma opção de *importar diretório*, mas geralmente o método mais simples é upload manual pela UI mesmo, já que permite checar e editar cada caption. Caso use o recurso de auto-caption (Florence) antes, revise e edite as legendas geradas para inserir a palavra-chave do LoRA conforme necessário. Por exemplo, se uma legenda automática veio como "A woman smiling in a snowy background", edite para "anafox smiling in a snowy background". **Todas as 18 legendas devem conter "anafox"** (ou o trigger escolhido).

   *Dica:* Mantenha as descrições consistentes. Se em algumas legendas você escreveu "uma pessoa de óculos" e em outras não mencionou os óculos, o modelo pode ficar confuso se óculos fazem parte ou não do conceito. Idealmente, mencione atributos variáveis apenas quando estiverem presentes, e atributos permanentes (ex: cor de cabelo, se for igual em todas fotos) em todas. No entanto, para identidades reais, geralmente não precisa listar muitas características – o modelo Flux já entende atributos visuais; o papel do LoRA aqui é vincular o **nome** àquela aparência. Portanto, não exagere detalhando demais cada imagem, foque no essencial.

4. **Verifique as configurações e inicie o treinamento (Seção *Training*):** Na terceira seção, o FluxGym exibirá o *treinamento pronto para iniciar*. Em muitos casos, ele mostra o comando exato do `train_network.py` (Kohya) que será executado, ou um resumo das configurações escolhidas, para conferência. Revise para garantir que está tudo correto – principalmente o número de steps esperados, learning rate, e que o modelo base selecionado é de fato o *flux1-dev*. Feito isso, clique no botão **Start Training**. O treinamento começará e você deverá ver na tela logs de progresso, incluindo possivelmente a perda (*loss*) sendo atualizada por step ou por epoch.

   * **Duração:** Treinar um LoRA no Flux com \~1800 passos pode levar algumas horas. Em um exemplo, treinar \~2000 passos numa GPU L4 levou \~4,5 horas. Com A100, pode ser mais rápido (talvez \~2 a 3 horas), mas isso varia conforme o batch size e outras otimizações. Tenha paciência e monitore periodicamente o output de log.
   * **Monitoramento e Overfitting:** Fique atento aos *logs* e quaisquer *imagens de amostra* geradas. O FluxGym permite configurar geração de **imagens de exemplo a cada N passos** (por exemplo, gerar uma prévia a cada 100 ou 200 steps). Se você habilitou isso (na seção LoRA Info ou Advanced, havia campos para prompts de amostra e intervalo), use essas imagens para avaliar o progresso do treinamento. No início, as amostras serão muito borradas, mas devem ir ganhando forma conforme os passos aumentam. **Interrompa o treinamento** (se possível) caso note algum desses sinais: 1) as imagens de amostra começaram a ficar **idênticas** às fotos de treino (isso indica *overfitting*, o modelo decorou os exemplos); 2) a perda de treinamento parou de cair e os outputs não melhoram; 3) surgem artefatos estranhos ou a qualidade piora depois de certo ponto (às vezes muito treino pode degradar resultados, especialmente se o learning rate estiver alto). Entretanto, de acordo com experiências, o Flux é **bastante resiliente e “difícil de overtreinar”** em comparação a modelos como SD1.5/SDXL. Mesmo com um dataset pequeno, ele tende a produzir bons resultados antes de começar a sobreajustar. Por via das dúvidas, fique de olho próximo do passo \~100 \* número de imagens (no nosso caso \~1800). Você pode salvar um checkpoint intermediário por volta disso (FluxGym/Kohya costuma salvar ao final de cada epoch ou em intervalos configurados) e comparar mais tarde com o final.
   * **Interatividade:** Não feche o Colab durante o treino. Se precisar pausar, evite; o ideal é deixar concluir. Caso enfrente *timeout* de sessão (desconexão), é provável que sua conta não seja Pro ou não esteja com a janela ativa – infelizmente o Colab grátis derruba sessões longas. Uma solução é dividir o treinamento em duas partes (salvar um modelo parcial e depois continuar), mas isso é avançado e não garantido dependendo do setup. O mais seguro é realmente usar Colab Pro e acompanhar o andamento.
   * **Possíveis erros:** Se o treinamento falhar no início com erro de *Out of Memory* no CUDA, verifique se não exagerou em algum parâmetro (por ex., muito steps de amostra simultânea, batch size alto, resolução incorreta). Reduza o batch ou desabilite geração de amostras para economizar VRAM. Se o erro mencionar algo sobre `image1`/`caption` não encontrado, pode ser um bug do workflow – geralmente causado por imagens não carregadas ou formatos errados; garanta que todas são PNG e estão com caminhos corretos.

   Uma vez iniciado, agora é aguardar até a finalização do processo de treino.

5. **Conclusão e salvamento do modelo LoRA:** Ao término do treinamento, o FluxGym deverá indicar que finalizou e mostrar o local do arquivo de saída (geralmente algo como `/content/fluxgym-Colab/outputs/<nome_do_lora>.safetensors`). Esse é o arquivo LoRA treinado, contendo apenas as diferenças aprendidas pelo modelo (tipicamente alguns dezenas de MB, dependendo do rank). É importante **salvá-lo fora do Colab** para não perdê-lo quando a sessão for reiniciada. Você pode usar a própria interface (algumas têm botão de download) ou no notebook Colab executar:

   ```python
   from google.colab import files
   files.download('/content/fluxgym-Colab/outputs/fluxLora_Ana.safetensors')
   ```

   Isso iniciará o download do arquivo para o seu computador. Alternativamente, mova o arquivo para sua pasta do Google Drive montada (ex: `!cp outputs/fluxLora_Ana.safetensors /content/drive/MyDrive/`) para guardar no Drive.

   *Dica:* Renomeie o arquivo se necessário para algo mais identificável (por exemplo, `Ana_FluxLoRA-1.safetensors`). Guarde também notas sobre quais configurações você usou (repeat, epochs, LR, etc.) para referência futura. Se planeja compartilhar o LoRA publicamente, lembre-se de que ele herda a **licença não-comercial do Flux Dev** (você não pode usá-lo comercialmente, e se publicar deve indicar a mesma licença).

   Até aqui, você completou o treinamento do LoRA! 🎉 Agora é hora de utilizar esse LoRA junto com o modelo base Flux para gerar novas imagens.

## 4. Uso de prompts para geração de imagens com o LoRA treinado

Com o LoRA treinado em mãos, o próximo passo é **gerar imagens** utilizando o modelo base Flux.1-Dev **acrescido do LoRA** que injeta a identidade visual desejada. Há diferentes formas de fazer isso; aqui abordaremos principalmente via **ComfyUI**, uma interface flexível para pipelines de difusão, que já possui nós customizados para suportar o Flux e aplicação de LoRAs. Você também pode usar outras UIs ou scripts, contanto que consiga carregar o modelo Flux com seus componentes e aplicar o LoRA sobre ele.

**1. Configurando o ambiente de geração (ComfyUI):** Se estiver no Colab, pode instalar e rodar o ComfyUI para geração. Caso contrário, pode usar uma instalação local. No Colab, por exemplo, você faria:

* Clonar o repositório do ComfyUI e instalar dependências:

  ```bash
  !git clone https://github.com/comfyanonymous/ComfyUI.git /content/ComfyUI
  %cd /content/ComfyUI
  !pip install xformers!=0.0.18 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
  ```
* Instalar nós customizados para Flux/gguf (um formato otimizado). Exemplo:

  ```bash
  %cd /content/ComfyUI/custom_nodes
  !git clone https://github.com/city96/ComfyUI-GGUF.git
  %cd ComfyUI-GGUF
  !pip install -r requirements.txt
  ```

  (Este nó customizado permite carregar modelos do tipo *GGUF* ou Flux no ComfyUI).
* Baixar/copiar os modelos para as pastas do ComfyUI:

  * Coloque o arquivo **flux1-dev-fp8.safetensors** na pasta `ComfyUI/models/checkpoints/` (checkpoint principal).
  * Coloque os encoders `clip_l.safetensors` e `t5xxl_fp8.safetensors` na pasta `ComfyUI/models/clip/`.
  * Coloque o VAE `ae.sft` em `ComfyUI/models/vae/`.
  * Coloque o seu arquivo LoRA `.safetensors` em `ComfyUI/models/lora/`.
* Inicie o ComfyUI em modo servidor no Colab e exponha via ngrok ou outro túnel (o Medium sugere uso do ngrok). Isso dará outra URL pública onde você acessará a interface do ComfyUI.

*Obs:* Os passos acima são um exemplo simplificado. Se estiver localmente, basta colocar os arquivos nas pastas do ComfyUI correspondentes e executar o `run_nvidia_gpu.bat` (Windows) ou script equivalente. Se usar Automatic1111 ou outra interface, consulte se há suporte ao Flux – por ser um modelo com dois encoders, nem todas UIs suportam diretamente. O ComfyUI atualmente é a opção mais confiável para Flux.

**2. Carregando o modelo Flux + LoRA na interface:** No ComfyUI, você precisará carregar o modelo Flux. Provavelmente será através de um **workflow pronto** ou montando manualmente os nós. O tutorial do Medium recomenda baixar um *workflow JSON* específico para Flux + LoRA e carregá-lo via botão *Load Workflow*. Esse workflow já conectaria os nós de checkpoint (Flux Dev), text encoders e aplicação de LoRA, assim você não precisa configurar tudo do zero. Certifique-se, após carregar o workflow, de selecionar nos nós os arquivos que você adicionou (por exemplo, escolher o checkpoint flux1-dev, selecionar seu LoRA no nó de LoRA Loader, apontar o VAE correto, etc.). Com tudo configurado, a interface deve ter algo como: nós de *Load Checkpoint*, *Load LoRA*, *VAEDecode*, etc., terminando em um *Save Image*.

Caso prefira fazer manualmente: adicione um nó de modelo (CheckpointLoader) e aponte para flux1-dev-fp8, conecte aos encoders (ou use um CheckpointLoader específico para Flux que já engloba encoders), depois um nó de Apply LoRA apontando para seu arquivo, então um nó de geração (Sampler) e por fim VAE Decode. Se isso for muito complexo, utilize o workflow fornecido pela comunidade para evitar erro de montagem.

**3. Compondo o prompt e parâmetros de geração:** Agora vem a parte criativa. Para gerar uma imagem do seu sujeito:

* **Escreva o prompt de texto incluindo a *trigger word*** do seu LoRA. *Exemplo:* se o trigger é "anafox", um prompt poderia ser: `"foto retrato de anafox sorrindo, usando chapéu de praia, fundo de paisagem tropical ao pôr-do-sol"`. Esse prompt pede ao modelo uma foto da *anafox* com chapéu na praia ao entardecer. Como o modelo base Flux foi treinado em **uma quantidade enorme de imagens reais**, ele sabe gerar ambientes realistas; o papel do LoRA é fazer com que a personagem central tenha a aparência daquela pessoa específica. **Não esqueça de incluir o nome/trigger exatamente** – se você deixar só "uma mulher sorrindo..." sem "anafox", o Flux gerará alguma mulher genérica, não a sua identidade personalizada.
* **Estilo do prompt:** O Flux.1-Dev tem capacidade de entender descrições detalhadas e sutis. Foi observado que ele *“gosta” de prompts longos e até poéticos* para tirar proveito do seu entendimento de texto. Portanto, sinta-se livre para elaborar na descrição: mencione cores, clima, iluminação, emoções, ambiente, etc. Quanto mais completo (dentro do razoável), mais o modelo terá para trabalhar e geralmente retorna imagens ricas. Por exemplo, em vez de "anafox na praia", você pode escrever "fotografia ao ar livre de anafox caminhando em uma praia deserta ao entardecer, céu alaranjado refletindo no mar, expressão serena". Flux tende a capturar bem esses detalhes.
* **Negative prompt:** Em modelos Stable Diffusion tradicionais, muitas vezes adicionamos *negative prompts* para evitar certos defeitos (como "deformidades, blur, etc."). Já o Flux-dev foi *distilado* de um modelo maior de forma que **não utiliza prompt negativo explicitamente para orientação**. Em outras palavras, o Flux foi treinado para gerar imagens de qualidade sem precisar que você diga o que *não* quer. Colocar um negative prompt não deve causar erro, mas seu efeito pode ser limitado. Recomenda-se focar no prompt positivo e, se necessário, você pode tentar negativas genéricas ("blurry, ugly, deformed") apenas para ver se nota diferença. Porém, muitos usuários relatam que não é muito necessário em Flux. Em vez disso, controle a saída ajustando o **CFG Scale**.
* **CFG Scale (Guidance):** O **CFG (Classifier-Free Guidance) Scale** é aquele parâmetro que equilibra fidelidade ao prompt versus qualidade geral. No Flux-dev, devido ao seu treinamento, valores moderados geralmente funcionam melhor. Um valor em torno de **3.5** (até 4 ou 5 no máximo) costuma ser ideal. Valores muito altos (ex: 7, 8, 12) podem degradar a imagem ou fazer o modelo perder a composição, já que o Flux já tem o "guidance" embutido em certa medida. Então experimente inicialmente **CFG = 3.5**. Se a imagem sair muito sem graça ou fora do assunto, suba um pouco; se vier com artefatos ou aparência "estourada", reduza.
* **Passos de amostragem (steps) e sampler:** Esses controlam o processo de difusão. O Flux geralmente atinge boa qualidade em **20 a 30 steps** usando samplers modernos (DPM++ 2M, Euler a, etc.), mas você pode usar **50 steps** para garantir mais nitidez se não se importar com um tempo um pouco maior. Acima de 50 steps raramente há ganhos significativos de qualidade para resoluções moderadas. Quanto ao **sampler**, escolha um compatível com difusão normal (Euler, Euler a, DDIM, DPM++...). O workflow do ComfyUI pode já definir um sampler. Você pode testar alguns – Euler **ancestral** costuma dar resultados fotográficos bons, DPM++ 2M Karras é ótimo para detalhes, etc.
* **Peso/força do LoRA:** Em algumas interfaces (como ComfyUI ou Automatic1111), é possível ajustar a **intensidade do LoRA** aplicado, geralmente via um slider ou campo "LoRA strength". Esse valor multiplica a influência do LoRA sobre o modelo base. Por padrão, considere **1.0 (100%)** como o valor normal. Testes com Flux mostraram que valores em torno de **1.0 a 1.3** produzem as melhores imagens, fiéis ao sujeito sem distorcer o estilo base. Acima disso, pode começar a **super-expor** o efeito (a imagem pode ficar com aspectos muito rígidos ou artefatos), e abaixo disso talvez o rosto não se pareça tanto. Então, inicialmente use peso **1.0**. Se achar que o resultado ainda não está capturando bem a pessoa, pode tentar 1.1 ou 1.2; se por outro lado a foto estiver parecendo *fake demais ou saturada*, poderia abaixar para 0.8. Esse ajuste fino ajuda a equilibrar contribuição do LoRA vs. criatividade do modelo base.
* **Outros parâmetros:** Mantenha a resolução de geração igual às imagens de treino (1024x1024) para começar – afinal, o modelo aprendeu nessa resolução. Depois você pode tentar gerar maior (com upscaling ou hires fix) ou variações de aspect ratio, mas inicialmente, 1024×1024 garante que o LoRA será plenamente eficaz. Use também o **mesmo VAE** do Flux (para cores ficarem corretas). Se o workflow não aplicou automaticamente, adicione o nó de VAE decode com `ae.sft`.

Feitas essas configurações, **clique para gerar** (no ComfyUI, adicionar à *queue* e depois *execute*, ou na interface que estiver usando). Aguarde a imagem ser produzida.

4. **Aprimore o prompt iterativamente:** Uma vez que a primeira imagem com o LoRA saia, avalie o resultado. Provavelmente, você verá o rosto da pessoa inserido na cena descrita. Se algo não estiver satisfatório, você pode refinar o prompt e tentar novamente. Por exemplo, se a face saiu parecida mas não perfeita, tente especificar melhor algum traço ("brown hair, green eyes", se for o caso real). Se o estilo da foto não agradou (digamos, ficou muito escuro), adicione termos como "bright lighting, high detail". Lembre-se que **prompts longos e bem esmiuçados tendem a funcionar bem no Flux** – diferente de alguns modelos que preferem frases telegráficas, o Flux suporta linguagem natural mais elaborada. Experimente também variar *seeds* (sementes aleatórias) para obter diferentes composições.

Em resumo, para gerar imagens com seu LoRA:

* **Carregue o modelo Flux Dev + componentes + LoRA** em uma pipeline de difusão compatível (recomendado ComfyUI).
* **Inclua a palavra-chave do LoRA no prompt** para invocar a identidade, junto com a descrição da cena desejada.
* **Use CFG \~3.5, 30-50 steps, sampler adequado**, mantendo resoluções próximas às de treino (1024×1024).
* **Ajuste a força do LoRA** se necessário, em torno de 1.0.
* **Capriche na descrição** (prompt detalhado) em vez de contar com negative prompt – o Flux foi otimizado para isso.

Agora você está pronto para criar imagens personalizadas! No próximo tópico, veremos como avaliar os resultados e garantir que o LoRA atenda às expectativas sem overfitting.

## 5. Validação do desempenho do LoRA (análise de resultados)

Depois de gerar várias imagens utilizando o LoRA treinado, é importante **validar o desempenho** dele – isto é, verificar se de fato incorporou a identidade visual corretamente e se o fez de maneira generalizada, sem overfitting. Aqui estão aspectos a observar e passos para validar seu LoRA de identidade facial:

* **Fidelidade da identidade:** Avalie se o **rosto/pessoa gerado** nas imagens realmente parece com a pessoa das fotos originais. Isso inclui formato do rosto, olhos, nariz, boca, cor/tom de pele, cabelo, etc. Experimente gerar **várias imagens com prompts diferentes** (mudando roupa, cenário, expressão) e veja se em todas a identidade se mantém reconhecível. Um bom LoRA de identidade deve **transferir o rosto consistentemente** para novas situações. Se em algumas imagens o rosto já não parece a mesma pessoa, ou fica genérico, o LoRA pode estar fraco (sub-treinado) – talvez precise de mais steps ou um rank maior. Por outro lado, se o rosto é idêntico em todas as imagens a ponto de parecer a *mesma foto copiada*, aí pode ser overfitting.

* **Verificação de generalização vs. memorização:** Um desafio de treinar com poucos exemplos é o modelo acabar **decorando detalhes específicos** das imagens de treino em vez de capturar apenas a essência. Para testar isso, gere intencionalmente imagens que *fogem um pouco* do que havia no dataset. Por exemplo, se nas 18 fotos a pessoa sempre estava sem óculos, tente um prompt colocando óculos nela; ou se sempre estava de cabelo solto, peça com rabo de cavalo; mude bastante o fundo, etc. O modelo conseguindo gerar essas variações **sem perder a cara da pessoa** indica boa generalização. Já se ele **ignorar suas instruções e reproduzir elementos das fotos originais**, pode ser um sintoma de overfitting. Por exemplo, houve um caso em que um LoRA de Flux treinado em imagens do Crash Bandicoot aprendeu até um detalhe mínimo – um pequeno laço na calça que aparecia em 10% das imagens – e passou a colocar esse laço em quase todas imagens geradas do personagem. Isso demonstra que o modelo ficou muito fiel ao dataset, talvez até demais em certos pormenores. No seu caso, verifique se o LoRA **está sempre gerando alguma coisa presente em todas as fotos** (ex: a mesma jaqueta que ele vestia no ensaio) mesmo quando você não pede. Se sim, ele pode ter se apegado a esse item como parte integral da identidade. Para mitigar, você poderia: adicionar variedade (mais fotos com roupas diferentes) ou nas gerações sempre especificar a roupa para sobrescrever isso. O equilíbrio entre **flexibilidade e fidelidade** é o objetivo – o LoRA deve reproduzir o rosto, mas não necessariamente a mesma roupa/cenário a não ser que você queira.

* **Qualidade e integridade das imagens geradas:** Observe aspectos técnicos das imagens geradas: resolução, nitidez, se há **artefatos estranhos ou distorções** em partes do corpo. Rostos saíram com olhos e mãos normais? (Mãos não são o foco, mas se incluiu corpo, cheque se não ficaram deformadas – isso às vezes acontece se o modelo foi sobrecarregado). Compare as imagens geradas usando o LoRA com imagens geradas **sem o LoRA** para o mesmo prompt (substitua o nome por algo genérico como "uma mulher" e veja). O ideal é que o LoRA agregue informação (o rosto específico) sem degradar a qualidade base da imagem. O Flux por si só já gera imagens realistas de alta qualidade; o LoRA não deve piorar isso. Se notar que com o LoRA as imagens ficaram mais borradas, ou sempre com um estilo meio fora do comum, pode ser indicativo de algum overfitting ou de que o LoRA está **sobrescrevendo demais o modelo base**. Nesse caso, tente diminuir a força do LoRA na geração (ex: 0.8) para ver se a qualidade melhora, ou re-treinar com rank menor.

* **Ajuste fino do peso do LoRA:** Já mencionado, mas faz parte da validação testar **diferentes intensidades** do LoRA aplicado nas gerações. Gere uma grade de imagens variando o peso: 0 (LoRA off), 0.5, 1.0, 1.5 por exemplo. Assim você visualiza desde o modelo puro até o LoRA exagerado. De 0 para 0.5 deve começar a aparecer semelhança, 1.0 é o alvo, 1.5 normalmente já seria too much (imagens possivelmente ficam estranhas). Isso confirma empiricamente qual intensidade é mais estável. A maioria dos LoRAs de identidade atinge seu sweet spot em 1.0 mesmo, mas não custa testar. Se em 1.0 ainda estiver fraco (rosto não tão parecido), talvez o LoRA não treinou o suficiente. Se em 1.0 já está introduzindo artefatos, talvez treinou demais. Essa validação empírica ajuda a decidir se vale a pena re-treinar.

* **Comparação multi-step (se disponível):** Alguns workflows (como o do Finetuners) geram previews do LoRA em diferentes checkpoints de treinamento para comparar. Se você tiver guardado modelos intermediários (por exemplo, em cada epoch), pode gerar a mesma imagem com cada um e ver em qual ponto ficou melhor. Às vezes o *melhor modelo* é antes do último epoch – se identificar isso, use aquele checkpoint em vez do final. O Flux (por sua robustez) tende a tolerar bem muito treino, mas casos como o do Crash Bandicoot mostraram queda de desempenho após \~3000 steps. Portanto, pode ser útil reavaliar se seus 1800 steps foram excessivos ou não. Pela nossa expectativa, 1800 deve ser OK.

* **Overfitting visível vs. latente:** Mesmo que as imagens geradas pareçam boas, pense no seguinte: o LoRA está **demais específico**? Por exemplo, se você colocar o nome da pessoa *sem mais nada no prompt*, a modelo gera sempre a mesma pose/fundo que tinha em alguma foto? Isso significaria que ele colou muito em uma imagem específica. O ideal é que com apenas o nome, o modelo possa renderizar a pessoa em configurações variadas (ainda que meio default). Faça esse teste: prompt somente "`anafox` portrait photo" e veja o que vem. Se for sempre extremamente parecido com uma determinada foto do dataset, talvez houve memorização daquela foto. Aí você pode precisar de mais variedade ou redução de epochs.

* **Iteração e melhorias:** A validação serve para **extrair insights** e refinar seu processo. Raramente um LoRA sai perfeito de primeira. Use o que você aprendeu das imagens geradas para, se necessário, **ajustar o treino**. Exemplos de ações pós-validação:

  * Se percebeu overfitting em certos detalhes, considere **remover ou substituir** no dataset imagens muito semelhantes entre si ou que contenham detalhes enganadores. Ou adicione **imagens novas** cobrindo situações ausentes (se todas fotos eram sorrindo, adicione algumas sérias, etc.).
  * Se a identidade não está forte o suficiente, talvez você precise **mais steps** de treino. Você poderia aumentar a repeat ou epochs (ex: de 10 para 12 epochs para chegar \~2160 steps). Sempre monitore para não ultrapassar o ponto ótimo.
  * Se a identidade está OK mas perdeu-se qualidade ou versatilidade, tente **reduzir a learning rate** e treinar de novo por mais passos – um aprendizado mais lento porém prolongado às vezes generaliza melhor que um rápido.
  * Ajuste do **Network Rank:** caso ache que o LoRA não consegue capturar algum detalhe (por ex., uma pinta de beleza ou formato específico de sorriso), um rank maior pode ajudar. Inversamente, se o LoRA está muito grande e específico, um rank menor pode fazê-lo mais maleável.
  * **Flux Dev2Pro:** Uma dica avançada vinda da comunidade: o Flux Dev sozinho pode ser difícil de finetunar perfeitamente por ser um modelo distilado. Uma técnica é usar o modelo **Flux-Dev2Pro** (uma versão do Flux melhor alinhada para treinamento, criada por terceiros) como base. Ele tende a dar LoRAs de melhor qualidade e depois você pode aplicar esses LoRAs no Flux-dev normal. Se notar instabilidades que não consegue resolver, vale experimentar essa abordagem numa futura iteração.

Lembre que **Flux é um modelo de ponta e bastante “permissivo”** – mesmo com um dataset pequeno e não perfeitamente ideal, costuma conseguir resultados sólidos. Portanto, se os seus resultados ainda não estão no nível desejado, persista nos ajustes e possivelmente você os alcançará. Valide sempre com vários prompts e cenários para ter certeza de que seu LoRA funciona de forma abrangente.

---

Seguindo este guia detalhado, você deverá conseguir **configurar o ambiente no Colab, preparar seu dataset de 18 imagens, treinar um LoRA no modelo Flux.1-Dev e usar prompts para gerar imagens personalizadas**, tudo isso evitando as armadilhas comuns (como overfitting) e otimizando para melhores resultados. Lembre-se de documentar seus experimentos e compartilhar apenas conforme as licenças permitirem. Boa sorte nas suas gerações com o **Flux LoRA** – aproveite o poder desse modelo para trazer sua identidade visual às criações de IA! 🚀

**Referências Utilizadas:**

* Andrew. *How to train Flux LoRA models*. Stable Diffusion Art, 31 Dez 2024.
* *Train your own FLUX LoRA model (Windows/Linux)*. StableDiffusionTutorials, 26 Set 2024.
* P. W. *Creating a Flux Dev LORA – Full Guide*. Reticulated.net, 27 Out 2024.
* *Flux LoRA Training with Flux Gym and ComfyUI on Colab*. Medium (Alaiy), 31 Jan 2025.
* *Training LoRA on Flux – Best Practices & Settings*. Finetuners.ai, 02 Set 2024.
* John Shi. *Why Flux LoRA So Hard to Train and How to Overcome It?*. Medium, 23 Ago 2024.
* Comentários em Stable Diffusion Art – Flux LoRA tutorial.
* Documentação do FluxGym (cocktailpeanut/fluxgym).
