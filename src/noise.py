import io
import numpy as np

from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

rng = np.random.default_rng(0)

def noise_cor(image: Image.Image) -> Image.Image:
  pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
  ganho = np.array([rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5)], dtype=np.float32)
  pixels = np.clip(pixels * ganho, 0, 255).astype(np.uint8)

  return Image.fromarray(pixels, "RGB")


def noise_escuro(image: Image.Image) -> Image.Image:
  fator = rng.uniform(0.3, 0.7)
  return ImageEnhance.Brightness(image).enhance(fator)


def noise_contraste(image: Image.Image) -> Image.Image:
  fator = rng.uniform(0.5, 0.9)
  return ImageEnhance.Contrast(image).enhance(fator)


def noise_oclusao(image: Image.Image) -> Image.Image:
  w, h = image.size
  bloqueada = image.convert("RGB").copy()

  area = max(1, round(w * h / 6))
  proporcao = rng.uniform(0.6, 1.25)
  largura = max(1, min(w, round((area * proporcao) ** 0.5)))
  altura = max(1, min(h, round((area / proporcao) ** 0.5)))

  x0 = int(rng.integers(0, w - largura + 1))
  y0 = int(rng.integers(0, h - altura + 1))
  x1 = x0 + largura - 1
  y1 = y0 + altura - 1
  cor = (int(rng.integers(150, 255)), int(rng.integers(150, 255)), int(rng.integers(150, 255)))
  ImageDraw.Draw(bloqueada).rectangle((x0, y0, x1, y1), fill=cor)

  return bloqueada


def noise_desfoque(image: Image.Image) -> Image.Image:
  raio = rng.uniform(0.5, 2.0)
  return image.filter(ImageFilter.GaussianBlur(radius=raio))


def noise_movimento(image: Image.Image) -> Image.Image:
  raio_x = rng.uniform(0.5, 2.0)
  raio_y = rng.uniform(0.5, 2.0)
  return image.filter(ImageFilter.GaussianBlur(radius=(raio_x, raio_y)))


def noise_resolucao(image: Image.Image) -> Image.Image:
  w, h = image.size
  fator = rng.uniform(0.3, 0.7)
  res_reducao = (max(1, round(w*fator)), max(1, round(h*fator)))

  reduzida = image.resize(res_reducao, Image.Resampling.NEAREST)
  return reduzida.resize(image.size, Image.Resampling.NEAREST)


def noise_ruido(image: Image.Image) -> Image.Image:
  sigma = rng.uniform(5.0, 20.0)
  pixels = np.asarray(image.convert("RGB"), dtype=np.int16)

  ruido = np.stack([
    np.asarray(Image.effect_noise(image.size, sigma), dtype=np.int16) - 128
    for _ in range(3)
  ], axis=-1)

  pixels = np.clip(pixels + ruido, 0, 255).astype(np.uint8)
  return Image.fromarray(pixels, "RGB")


def noise_sal_pimenta(image: Image.Image) -> Image.Image:
  pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
  probabilidade = rng.uniform(0.02, 0.1)
  sorteio = rng.random(pixels.shape[:2])

  pixels[sorteio < probabilidade / 2] = 0
  pixels[(sorteio >= probabilidade / 2) & (sorteio < probabilidade)] = 255

  return Image.fromarray(pixels, "RGB")


def noise_jpeg(image: Image.Image) -> Image.Image:
  qualidade = int(rng.integers(10, 51))

  with io.BytesIO() as buffer:
    image.convert("RGB").save(buffer, format="JPEG", quality=qualidade)
    buffer.seek(0)

    with Image.open(buffer) as comprimida:
      return comprimida.copy()


def noise_nenhuma(image: Image.Image) -> Image.Image:
  return image


NOISE = {"cor":noise_cor,
         "escuro":noise_escuro,
         "contraste":noise_contraste, # baixo contraste
         "oclusao":noise_oclusao,
         "desfoque":noise_desfoque,
         "movimento":noise_movimento,
         "resolucao":noise_resolucao, # baixa resolucao
         "ruido":noise_ruido, # ruido gaussiano
         "sal_pimenta":noise_sal_pimenta,
         "jpeg":noise_jpeg,
         "nenhuma":noise_nenhuma,
         }