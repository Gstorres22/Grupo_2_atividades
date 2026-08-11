package com.example.clone_carterinha_digital.data

/** Todos os campos que o aluno pode preencher na carteirinha. */
data class StudentCard(
    val name: String,
    val course: String,
    val cpf: String,
    val rg: String,
    val birth: String,
    val rm: String,
    val validity: String
)
