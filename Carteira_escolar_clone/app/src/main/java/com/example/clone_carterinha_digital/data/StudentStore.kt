package com.example.clone_carterinha_digital.data

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import com.example.clone_carterinha_digital.R
import java.io.File
import java.io.FileOutputStream

/**
 * Guarda os dados da carteirinha no proprio aparelho.
 *
 * Os textos ficam em SharedPreferences e a foto e copiada para a pasta privada
 * do app. Nao ha banco de dados nem servidor: fechando o app, tudo continua salvo.
 */
object StudentStore {

    private const val PREFS_NAME = "carteirinha"
    private const val PHOTO_NAME = "foto_aluno.jpg"

    /** Maior lado da foto guardada, em pixels. Evita salvar imagens gigantes. */
    private const val MAX_PHOTO_SIZE = 1200

    private const val KEY_NAME = "nome"
    private const val KEY_COURSE = "curso"
    private const val KEY_CPF = "cpf"
    private const val KEY_RG = "rg"
    private const val KEY_BIRTH = "nascimento"
    private const val KEY_RM = "rm"
    private const val KEY_VALIDITY = "validade"

    private fun prefs(context: Context) =
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun load(context: Context): StudentCard {
        val prefs = prefs(context)
        val res = context.resources
        return StudentCard(
            name = prefs.getString(KEY_NAME, null) ?: res.getString(R.string.default_name),
            course = prefs.getString(KEY_COURSE, null) ?: res.getString(R.string.default_course),
            cpf = prefs.getString(KEY_CPF, null) ?: res.getString(R.string.default_cpf),
            rg = prefs.getString(KEY_RG, null) ?: res.getString(R.string.default_rg),
            birth = prefs.getString(KEY_BIRTH, null) ?: res.getString(R.string.default_birth),
            rm = prefs.getString(KEY_RM, null) ?: res.getString(R.string.default_rm),
            validity = prefs.getString(KEY_VALIDITY, null)
                ?: res.getString(R.string.default_validity)
        )
    }

    fun save(context: Context, card: StudentCard) {
        prefs(context).edit()
            .putString(KEY_NAME, card.name)
            .putString(KEY_COURSE, card.course)
            .putString(KEY_CPF, card.cpf)
            .putString(KEY_RG, card.rg)
            .putString(KEY_BIRTH, card.birth)
            .putString(KEY_RM, card.rm)
            .putString(KEY_VALIDITY, card.validity)
            .apply()
    }

    // ----------------------------------------------------------------- foto

    private fun photoFile(context: Context) =
        File(context.applicationContext.filesDir, PHOTO_NAME)

    fun loadPhoto(context: Context): Bitmap? {
        val file = photoFile(context)
        if (!file.exists()) return null
        return runCatching { BitmapFactory.decodeFile(file.absolutePath) }.getOrNull()
    }

    fun savePhoto(context: Context, bitmap: Bitmap): Boolean = runCatching {
        FileOutputStream(photoFile(context)).use { output ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 92, output)
        }
        true
    }.getOrDefault(false)

    fun deletePhoto(context: Context) {
        runCatching { photoFile(context).delete() }
    }

    /**
     * Le a imagem escolhida na galeria, corrige a rotacao gravada no EXIF
     * (senao fotos tiradas em pe aparecem deitadas) e reduz o tamanho.
     */
    fun decodePickedPhoto(context: Context, uri: Uri): Bitmap? = runCatching {
        val resolver = context.contentResolver

        val orientation = resolver.openInputStream(uri)?.use { input ->
            ExifInterface(input)
                .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        } ?: ExifInterface.ORIENTATION_NORMAL

        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }

        val options = BitmapFactory.Options().apply {
            inSampleSize = sampleSize(bounds.outWidth, bounds.outHeight)
        }
        val decoded = resolver.openInputStream(uri)
            ?.use { BitmapFactory.decodeStream(it, null, options) }
            ?: return null

        applyOrientation(decoded, orientation)
    }.getOrNull()

    private fun sampleSize(width: Int, height: Int): Int {
        if (width <= 0 || height <= 0) return 1
        var sample = 1
        while (maxOf(width, height) / (sample * 2) >= MAX_PHOTO_SIZE) sample *= 2
        return sample
    }

    private fun applyOrientation(bitmap: Bitmap, orientation: Int): Bitmap {
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            else -> return bitmap
        }
        return runCatching {
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        }.getOrDefault(bitmap)
    }
}
