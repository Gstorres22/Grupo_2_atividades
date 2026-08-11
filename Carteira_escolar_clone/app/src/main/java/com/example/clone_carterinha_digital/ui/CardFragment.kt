package com.example.clone_carterinha_digital.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.animation.AccelerateInterpolator
import android.view.animation.DecelerateInterpolator
import android.widget.ImageView
import android.widget.Toast
import androidx.fragment.app.Fragment
import com.example.clone_carterinha_digital.R
import com.example.clone_carterinha_digital.data.StudentStore
import com.example.clone_carterinha_digital.databinding.FragmentCardBinding

/**
 * Tela inicial: mostra a carteirinha. O botao de girar troca entre frente e verso.
 * Os botoes de QR Code e compartilhar sao apenas visuais nesta versao.
 */
class CardFragment : Fragment() {

    private var _binding: FragmentCardBinding? = null
    private val binding get() = _binding!!

    private var showingFront = true
    private var flipping = false

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentCardBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Sem isso a animacao de virar fica com uma perspectiva exagerada.
        val cameraDistance = 9000f * resources.displayMetrics.density
        binding.cardFront.root.cameraDistance = cameraDistance
        binding.cardBack.root.cameraDistance = cameraDistance
        binding.cardBack.root.visibility = View.GONE

        binding.btnFlip.setOnClickListener { flip() }
        binding.btnQr.setOnClickListener { showComingSoon() }
        binding.btnShare.setOnClickListener { showComingSoon() }
    }

    override fun onResume() {
        super.onResume()
        // Recarrega ao voltar da tela de edicao, para mostrar o que foi salvo.
        bindCard()
    }

    override fun onDestroyView() {
        binding.cardFront.root.animate().cancel()
        binding.cardBack.root.animate().cancel()
        super.onDestroyView()
        _binding = null
    }

    private fun bindCard() {
        val card = StudentStore.load(requireContext())

        binding.cardFront.tvName.text = card.name
        binding.cardFront.tvCourse.text = card.course

        binding.cardBack.tvCpf.text = card.cpf
        binding.cardBack.tvRg.text = card.rg
        binding.cardBack.tvBirth.text = card.birth
        binding.cardBack.tvRm.text = card.rm
        binding.cardBack.tvValidity.text = card.validity

        val photo = StudentStore.loadPhoto(requireContext())
        val photoView = binding.cardFront.imgPhoto
        if (photo != null) {
            photoView.scaleType = ImageView.ScaleType.CENTER_CROP
            photoView.setImageBitmap(photo)
        } else {
            photoView.scaleType = ImageView.ScaleType.CENTER_INSIDE
            photoView.setImageResource(R.drawable.ic_photo_placeholder)
        }
    }

    /** Vira a carteirinha com uma animacao de giro em 3D. */
    private fun flip() {
        if (flipping) return
        flipping = true

        val leaving = if (showingFront) binding.cardFront.root else binding.cardBack.root
        val entering = if (showingFront) binding.cardBack.root else binding.cardFront.root

        leaving.animate()
            .rotationY(90f)
            .setDuration(160L)
            .setInterpolator(AccelerateInterpolator())
            .withEndAction {
                if (_binding == null) return@withEndAction
                leaving.visibility = View.GONE
                leaving.rotationY = 0f
                entering.rotationY = -90f
                entering.visibility = View.VISIBLE
                entering.animate()
                    .rotationY(0f)
                    .setDuration(160L)
                    .setInterpolator(DecelerateInterpolator())
                    .withEndAction {
                        showingFront = !showingFront
                        flipping = false
                    }
                    .start()
            }
            .start()
    }

    private fun showComingSoon() {
        Toast.makeText(requireContext(), R.string.coming_soon, Toast.LENGTH_SHORT).show()
    }
}
