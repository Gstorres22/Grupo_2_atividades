// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
    alias(libs.plugins.android.application) apply false
}

/*
 * Este projeto fica dentro da pasta do OneDrive. Durante a sincronizacao o OneDrive
 * segura arquivos abertos e o Gradle falha com erros do tipo
 * "Unable to delete directory ...\app\build\intermediates\...".
 *
 * Para evitar isso, os arquivos temporarios de compilacao sao gravados fora do OneDrive,
 * em C:\Users\<usuario>\.gradle-builds\Clone_carterinha_digital.
 * O codigo-fonte continua normalmente aqui no projeto/GitHub.
 *
 * O APK gerado passa a ficar em:
 *   C:\Users\<usuario>\.gradle-builds\Clone_carterinha_digital\app\outputs\apk\debug\
 */
val buildRoot = File(System.getProperty("user.home"), ".gradle-builds/${rootProject.name}")

rootProject.layout.buildDirectory.set(File(buildRoot, "root"))
subprojects {
    layout.buildDirectory.set(File(buildRoot, name))
}
