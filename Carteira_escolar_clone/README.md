# ID Digital FIAP — clone da carteirinha digital

App Android que reproduz a carteirinha digital do aluno FIAP. Diferente do app original,
aqui **o próprio aluno preenche** os dados e a foto, direto no celular.

Projeto do Grupo 2 para o bootcamp de melhoria de aplicativos.

## O que o app faz

| Tela | Descrição |
|---|---|
| **Home** | Mostra a carteirinha. O botão de girar alterna frente ↔ verso com animação de virada. |
| **ID Digital** | Formulário onde o aluno edita foto, nome, curso, CPF, RG, nascimento, RM e validade. |

Os botões de QR Code e compartilhar são **decorativos** nesta versão (mostram "Disponível em breve").

Não há banco de dados nem login: os textos ficam em `SharedPreferences` e a foto é copiada
para a pasta privada do app. Fechando o app, tudo continua salvo.

## Como a carteirinha é desenhada

O ponto central do projeto está em [`CardStageLayout`](app/src/main/java/com/example/clone_carterinha_digital/widget/CardStageLayout.kt).

A carteirinha é um cartão **deitado**, desenhado num tamanho fixo de projeto de
**534 × 306 dp** ([`dimens.xml`](app/src/main/res/values/dimens.xml)). A `CardStageLayout`
mede os filhos sempre nesse tamanho, gira 90° e escala para caber na tela — ocupando
**84,9% da largura** (proporção medida nos prints originais).

Isso permite que todas as medidas dos layouts sejam dp fixos tirados da referência,
sem deixar de se adaptar a telas diferentes:

- [`view_card_front.xml`](app/src/main/res/layout/view_card_front.xml) — foto, logo, nome e curso
- [`view_card_back.xml`](app/src/main/res/layout/view_card_back.xml) — logo, linha divisória e a coluna de dados

### Medidas extraídas dos prints originais

Referência tirada de um print 720×1560. Valores no espaço de projeto do cartão (534 × 306 dp):

| Elemento | Posição / tamanho |
|---|---|
| Cartão | 534 × 306 dp, cantos de 16 dp, 84,9% da largura da tela |
| Foto (frente) | início 36 dp, topo 74 dp, 131 × 153 dp |
| Logo FIAP (frente) | início 193 dp, topo 86 dp, 150 × 40 dp |
| Nome do aluno | 18 dp, `letterSpacing` 0.075, Poppins Bold |
| Nome do curso | 12 dp, `letterSpacing` 0.07, entrelinha 0.72 |
| Logo FIAP (verso) | início 106 dp, topo 137 dp, 110 × 30 dp |
| Linha divisória (verso) | início 238 dp, topo 106 dp, 1,5 × 120 dp |
| Dados (verso) | início 260 dp, topo 112 dp, linhas de 19 dp, texto 13 dp |

## Decisões que não são óbvias no código

**Textos do cartão em `dp`, não em `sp`.** A carteirinha tem proporção fixa, como uma imagem.
Se usasse `sp`, o ajuste de tamanho de fonte do sistema deformaria o cartão — num aparelho
com fonte em 0,8 os textos encolhiam 20%.

**Logo FIAP como vetor.** [`ic_fiap_logo.xml`](app/src/main/res/drawable/ic_fiap_logo.xml)
foi redesenhado traço a traço a partir do print: o "F" com a barra do meio solta da haste,
o "A" com a quebra na perna direita e o "P" com a barra inferior destacada.

**Pasta de build fora do OneDrive.** O projeto fica dentro do OneDrive, que trava arquivos
durante a sincronização e faz o Gradle falhar com `Unable to delete directory`. O
[`build.gradle.kts`](build.gradle.kts) da raiz redireciona os arquivos temporários para
`~/.gradle-builds/`. O APK gerado fica em
`~/.gradle-builds/Clone_carterinha_digital/app/outputs/apk/debug/`.

**Valores iniciais são placeholders.** `NOME DO ALUNO`, `000.000.000-00` etc. — dados reais
nunca ficam no código, só no aparelho de quem usa o app.

## Fonte

Poppins (SIL Open Font License 1.1), em [`app/src/main/res/font/`](app/src/main/res/font/).
Licença completa em [`POPPINS_OFL.txt`](POPPINS_OFL.txt).

## Como rodar

1. Abrir a pasta `Carteira_escolar_clone` no Android Studio
2. Esperar o Gradle sync terminar
3. Escolher o aparelho no seletor do topo e clicar em ▶

Requisitos: `minSdk 24`, `compileSdk 37`, Java 11.
