package com.example.clone_carterinha_digital.ui

import android.graphics.Bitmap
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import com.example.clone_carterinha_digital.R
import com.example.clone_carterinha_digital.data.StudentCard
import com.example.clone_carterinha_digital.data.StudentStore
import com.example.clone_carterinha_digital.databinding.FragmentEditBinding

/**
 * Tela "ID Digital": o aluno preenche os dados e escolhe a foto da carteirinha.
 * Nada e gravado ate ele tocar em "Salvar".
 */
class EditFragment : Fragment() {

    private var _binding: FragmentEditBinding? = null
    private val binding get() = _binding!!

    /** Foto escolhida agora, ainda nao gravada. */
    private var pendingPhoto: Bitmap? = null

    /** O aluno pediu para apagar a foto atual. */
    private var photoMarkedForRemoval = false

    private val pickPhoto =
        registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
            if (uri == null || _binding == null) return@registerForActivityResult

            val bitmap = StudentStore.decodePickedPhoto(requireContext(), uri)
            if (bitmap == null) {
                Toast.makeText(requireContext(), R.string.edit_photo_error, Toast.LENGTH_LONG)
                    .show()
                return@registerForActivityResult
            }

            pendingPhoto = bitmap
            photoMarkedForRemoval = false
            showPhoto(bitmap)
        }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentEditBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Deixa o conteudo subir acima do teclado sem ficar escondido.
        ViewCompat.setOnApplyWindowInsetsListener(binding.editScroll) { v, insets ->
            val ime = insets.getInsets(WindowInsetsCompat.Type.ime()).bottom
            v.updatePadding(bottom = ime)
            insets
        }

        fillFields()

        binding.btnChoosePhoto.setOnClickListener {
            pickPhoto.launch(
                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
            )
        }
        binding.btnRemovePhoto.setOnClickListener {
            pendingPhoto = null
            photoMarkedForRemoval = true
            showPlaceholder()
        }
        binding.btnSave.setOnClickListener { saveAndGoBack() }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    private fun fillFields() {
        val card = StudentStore.load(requireContext())
        binding.inputName.setText(card.name)
        binding.inputCourse.setText(card.course)
        binding.inputCpf.setText(card.cpf)
        binding.inputRg.setText(card.rg)
        binding.inputBirth.setText(card.birth)
        binding.inputRm.setText(card.rm)
        binding.inputValidity.setText(card.validity)

        val photo = StudentStore.loadPhoto(requireContext())
        if (photo != null) showPhoto(photo) else showPlaceholder()
    }

    private fun showPhoto(bitmap: Bitmap) {
        binding.imgPhotoPreview.scaleType = ImageView.ScaleType.CENTER_CROP
        binding.imgPhotoPreview.setImageBitmap(bitmap)
    }

    private fun showPlaceholder() {
        binding.imgPhotoPreview.scaleType = ImageView.ScaleType.CENTER_INSIDE
        binding.imgPhotoPreview.setImageResource(R.drawable.ic_photo_placeholder)
    }

    private fun saveAndGoBack() {
        val context = requireContext()

        StudentStore.save(
            context,
            StudentCard(
                name = binding.inputName.text.textOr(R.string.default_name),
                course = binding.inputCourse.text.textOr(R.string.default_course),
                cpf = binding.inputCpf.text.textOr(R.string.default_cpf),
                rg = binding.inputRg.text.textOr(R.string.default_rg),
                birth = binding.inputBirth.text.textOr(R.string.default_birth),
                rm = binding.inputRm.text.textOr(R.string.default_rm),
                validity = binding.inputValidity.text.textOr(R.string.default_validity)
            )
        )

        pendingPhoto?.let { StudentStore.savePhoto(context, it) }
        if (photoMarkedForRemoval) StudentStore.deletePhoto(context)

        Toast.makeText(context, R.string.edit_saved, Toast.LENGTH_SHORT).show()

        val navController = findNavController()
        if (!navController.popBackStack(R.id.cardFragment, false)) {
            navController.navigate(R.id.cardFragment)
        }
    }

    /** Campo vazio volta para o valor de exemplo, para a carteirinha nunca ficar em branco. */
    private fun CharSequence?.textOr(fallbackRes: Int): String {
        val value = this?.toString()?.trim().orEmpty()
        return value.ifEmpty { getString(fallbackRes) }
    }
}
