package com.example.clone_carterinha_digital.widget

import android.content.Context
import android.util.AttributeSet
import android.widget.FrameLayout
import com.example.clone_carterinha_digital.R
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * "Palco" da carteirinha.
 *
 * A carteirinha e desenhada deitada, num tamanho fixo de projeto (534 x 306 dp),
 * exatamente como um cartao de verdade. Esta View pega esse desenho, gira 90 graus
 * e reduz/aumenta para caber no espaco disponivel da tela.
 *
 * A vantagem: todas as medidas dos layouts da carteirinha ficam em dp fixos
 * (tiradas das imagens de referencia) e continuam proporcionais em qualquer celular.
 */
class CardStageLayout @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : FrameLayout(context, attrs, defStyleAttr) {

    private companion object {
        /**
         * Quanto da largura da tela a carteirinha ocupa. Medido na imagem de
         * referencia (611 de 720 px). Usar fracao em vez de margem fixa mantem a
         * mesma proporcao em telas estreitas e largas.
         */
        const val CARD_WIDTH_FRACTION = 0.849f
    }

    /** Largura do cartao deitado, em pixels. */
    private val designWidth: Int
        get() = resources.getDimensionPixelSize(R.dimen.card_design_width)

    /** Altura do cartao deitado, em pixels. */
    private val designHeight: Int
        get() = resources.getDimensionPixelSize(R.dimen.card_design_height)

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        // Os filhos sempre recebem o tamanho de projeto; a adaptacao vem da escala.
        val childWidthSpec = MeasureSpec.makeMeasureSpec(designWidth, MeasureSpec.EXACTLY)
        val childHeightSpec = MeasureSpec.makeMeasureSpec(designHeight, MeasureSpec.EXACTLY)
        for (i in 0 until childCount) {
            getChildAt(i).measure(childWidthSpec, childHeightSpec)
        }
        setMeasuredDimension(
            resolveSize(MeasureSpec.getSize(widthMeasureSpec), widthMeasureSpec),
            resolveSize(MeasureSpec.getSize(heightMeasureSpec), heightMeasureSpec)
        )
    }

    override fun onLayout(changed: Boolean, left: Int, top: Int, right: Int, bottom: Int) {
        val availableWidth = width - paddingLeft - paddingRight
        val availableHeight = height - paddingTop - paddingBottom
        if (availableWidth <= 0 || availableHeight <= 0) return

        // Depois de girar 90 graus, a largura na tela passa a ser a altura do projeto
        // e vice-versa. Por isso os termos ficam cruzados aqui.
        val targetWidth = min(availableWidth.toFloat(), width * CARD_WIDTH_FRACTION)
        val scale = min(
            targetWidth / designHeight,
            availableHeight.toFloat() / designWidth
        )

        val centerX = paddingLeft + availableWidth / 2f
        val centerY = paddingTop + availableHeight / 2f
        val childLeft = (centerX - designWidth / 2f).roundToInt()
        val childTop = (centerY - designHeight / 2f).roundToInt()

        for (i in 0 until childCount) {
            val child = getChildAt(i)
            child.layout(childLeft, childTop, childLeft + designWidth, childTop + designHeight)
            child.pivotX = designWidth / 2f
            child.pivotY = designHeight / 2f
            child.rotation = 90f
            child.scaleX = scale
            child.scaleY = scale
        }
    }
}
