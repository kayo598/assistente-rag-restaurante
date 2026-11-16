import json
import random
import re
import shutil

import pytest

from rag_restaurante.assistente import Assistente
from rag_restaurante.avaliacao import GABARITO_COLOQUIAL, GABARITO_PADRAO
from rag_restaurante.busca import IndiceBM25, carregar_indice, palavras, tokenizar
from rag_restaurante.trechos import (
    PASTA_PADRAO,
    ErroDocumentos,
    Trecho,
    carregar_pasta,
    carregar_sinonimos,
    dividir_markdown,
)


def test_divide_por_secao():
    md = "# Doc\n\n## Um\n\nTexto um.\n\n## Dois\n\nTexto dois."
    assert [(t.secao, t.texto) for t in dividir_markdown(md, "x.md")] == [("Um", "Texto um."), ("Dois", "Texto dois.")]


def test_texto_antes_da_primeira_secao_nao_se_perde():
    trechos = dividir_markdown("# Horários\n\nAbrimos às 18h.\n\n## Feriados\n\nFechado.", "x.md")
    assert [(t.secao, t.texto) for t in trechos] == [("Horários", "Abrimos às 18h."), ("Feriados", "Fechado.")]


def test_subsecao_fica_dentro_da_secao():
    trechos = dividir_markdown("## A\n\ntexto a\n\n### B\n\ntexto b", "x.md")
    assert len(trechos) == 1 and "### B" in trechos[0].texto


def test_quebra_secao_longa_por_paragrafo():
    corpo = "\n\n".join(" ".join(["palavra"] * 50) for _ in range(3))
    trechos = dividir_markdown(f"## Longa\n\n{corpo}", "x.md", max_palavras=120)
    assert len(trechos) == 2 and all(t.secao == "Longa" for t in trechos)


def test_documento_salvo_no_windows(tmp_path):
    (tmp_path / "w.md").write_bytes("## Seção\n\nCafé com pão".encode("cp1252"))
    assert carregar_pasta(tmp_path)[0].texto == "Café com pão"


def test_documento_com_bom(tmp_path):
    (tmp_path / "b.md").write_bytes("﻿## Seção\n\nTexto".encode())
    assert carregar_pasta(tmp_path)[0].secao == "Seção"


@pytest.mark.parametrize("subpasta, mensagem", [("nao_existe", "não existe"), ("vazia", "nenhum documento")])
def test_pasta_invalida_tem_mensagem_clara(tmp_path, subpasta, mensagem):
    (tmp_path / "vazia").mkdir()
    with pytest.raises(ErroDocumentos, match=mensagem):
        carregar_pasta(tmp_path / subpasta)


def test_documentos_do_pacote():
    trechos = carregar_pasta()
    assert len(trechos) == 17
    assert {t.fonte for t in trechos} == {"alergenos.md", "cardapio.md", "entrega.md", "politicas.md"}


def test_tokenizar_normaliza_acento_plural_e_radical():
    assert tokenizar("Pizzas") == tokenizar("pizza")
    assert tokenizar("celíaco") == tokenizar("celíaca")
    assert tokenizar("cancelar") == tokenizar("cancelamento")
    assert tokenizar("o de para") == []


@pytest.mark.parametrize(
    "pergunta, secao",
    [
        ("Abre segunda-feira?", "Horário de funcionamento"),
        ("Posso cancelar o pedido?", "Cancelamento"),
        ("Vocês vendem cerveja?", "Bebidas"),
    ],
)
def test_busca_encontra_secao_certa(indice, pergunta, secao):
    assert indice.buscar(pergunta)[0].trecho.secao == secao


def test_indice_vazio_e_recusado():
    with pytest.raises(ValueError, match="pelo menos um trecho"):
        IndiceBM25([])


def test_indice_com_trecho_sem_palavras():
    indice = IndiceBM25([Trecho("x.md", "A", "de da do"), Trecho("x.md", "B", "pizza")])
    assert [r.trecho.secao for r in indice.buscar("pizza")] == ["B"]


def test_fuzz_busca_nunca_quebra(indice):
    gerador = random.Random(3)
    alfabeto = "abcçãéí ,.;!?\n\t\"'[]{}<>/\\😀\x000123456789-_"
    for _ in range(3000):
        pergunta = "".join(gerador.choice(alfabeto) for _ in range(gerador.randint(0, 60)))
        for resultado in indice.buscar(pergunta):
            assert resultado.score > 0


def test_fuzz_markdown_aleatorio_nunca_quebra():
    gerador = random.Random(4)
    pedacos = ["# ", "## ", "### ", "\n", "\n\n", "texto", "Seção", " ", "-", "😀"]
    for _ in range(2000):
        md = "".join(gerador.choice(pedacos) for _ in range(gerador.randint(0, 40)))
        for t in dividir_markdown(md, "x.md"):
            assert t.texto.strip()


def test_documento_ilegivel_tem_mensagem_clara(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("## A\n\ntexto", encoding="utf-8")

    def negar(self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(type(tmp_path), "read_bytes", negar)
    with pytest.raises(ErroDocumentos, match=r"não consegui ler a\.md: Permission denied"):
        carregar_pasta(tmp_path)


def test_relevancia_fica_entre_0_e_1(indice):
    for pergunta in ["Abre segunda?", "taxa de entrega", "pizza calabresa preço", "xyz"]:
        for resultado in indice.buscar(pergunta):
            assert 0 < indice.relevancia(pergunta, resultado) <= 1


def test_palavra_que_nao_existe_nos_documentos_derruba_a_relevancia(indice):
    resultado = indice.buscar("pizza")[0]
    assert indice.relevancia("pizza", resultado) > indice.relevancia("pizza helicóptero submarino", resultado)


@pytest.mark.parametrize("arquivos", [["entrega.md"], ["entrega.md", "politicas.md"]])
def test_colecoes_pequenas_nao_recusam_pergunta_boa(tmp_path, arquivos):
    """Com o limiar fixo antigo (1,5), uma coleção de um documento recusava perguntas boas."""
    for nome in arquivos:
        shutil.copy(PASTA_PADRAO / nome, tmp_path / nome)
    pequeno = IndiceBM25(carregar_pasta(tmp_path))
    gabarito = json.loads(GABARITO_PADRAO.read_text(encoding="utf-8"))
    boas = [c["pergunta"] for c in gabarito["com_resposta"] if c["fonte"] in arquivos]
    assert boas and not any(Assistente(pequeno).responder(p).recusa for p in boas)


def test_segundo_titulo_nao_rouba_o_texto_da_secao_anterior():
    """Antes, "texto um" era citado como se fosse da seção B."""
    trechos = dividir_markdown("# A\n## s1\ntexto um\n# B\ntexto dois", "x.md")
    assert [(t.secao, t.texto) for t in trechos] == [("s1", "texto um"), ("B", "texto dois")]


def test_fuzz_cada_linha_fica_no_trecho_da_secao_certa():
    """Para qualquer markdown, cada linha de texto aparece uma vez, no trecho do título mais próximo acima."""
    gerador = random.Random(5)
    for _ in range(1500):
        linhas, esperado, secao = [], [], "x"
        for contador in range(gerador.randint(1, 12)):
            tipo = gerador.choice(["#", "##", "texto", "texto", "vazia"])
            if tipo in ("#", "##"):
                secao = f"T{contador}"
                linhas.append(f"{tipo} {secao}")
            elif tipo == "texto":
                linhas.append(f"linha{contador}")
                esperado.append((secao, f"linha{contador}"))
            else:
                linhas.append("")
        obtido = [(t.secao, linha) for t in dividir_markdown("\n".join(linhas), "x.md") for linha in t.texto.split()]
        assert obtido == esperado, "\n".join(linhas)


def sinonimos_em(tmp_path, conteudo, encoding="utf-8"):
    (tmp_path / "a.md").write_text(
        "## Taxa\n\nTaxa de entrega: R$ 5,00.\n\n## Bebidas\n\nPelo delivery.", encoding="utf-8"
    )
    (tmp_path / "sinonimos.json").write_text(conteudo, encoding=encoding)
    return tmp_path


def test_sinonimos_sao_opcionais(tmp_path):
    (tmp_path / "a.md").write_text("## A\n\ntexto", encoding="utf-8")
    assert carregar_sinonimos(tmp_path) == {}


@pytest.mark.parametrize(
    "conteudo, mensagem",
    [
        ("{x", "não é um JSON válido"),
        ('["frete"]', "precisa ser um objeto"),
        ('{"frete": 3}', "precisa ser um texto"),
        ('{"frete": " "}', "precisa ser um texto"),
        ('{"vale refeição": "vr"}', "uma palavra só"),
        ('{"coca-cola": "refrigerante"}', "uma palavra só"),
    ],
)
def test_sinonimos_invalidos_tem_mensagem_clara(tmp_path, conteudo, mensagem):
    with pytest.raises(ErroDocumentos, match=mensagem):
        carregar_sinonimos(sinonimos_em(tmp_path, conteudo))


def test_sinonimos_salvos_no_windows(tmp_path):
    pasta = sinonimos_em(tmp_path, '{"Fréte": "taxa de entrega"}', encoding="cp1252")
    indice = carregar_indice(pasta)
    assert indice.buscar("qual o frete?", 1)[0].trecho.secao == "Taxa"  # chave normalizada: sem acento e minúscula


def test_palavra_desconhecida_e_trocada_pelo_sinonimo(tmp_path):
    """'frete' não existe nos documentos: somada, derrubaria a relevância. Ela é trocada."""
    indice = carregar_indice(sinonimos_em(tmp_path, '{"frete": "taxa de entrega"}'))
    assert indice.consulta("valor do frete") == tokenizar("valor taxa de entrega")
    sem = IndiceBM25(indice.trechos)
    assert not sem.buscar("frete") and indice.buscar("frete")[0].trecho.secao == "Taxa"


def test_palavra_conhecida_fica_e_ganha_o_sinonimo(tmp_path):
    """'delivery' aparece nos documentos: trocar perderia esse casamento."""
    indice = carregar_indice(sinonimos_em(tmp_path, '{"delivery": "entrega"}'))
    assert indice.consulta("delivery") == tokenizar("delivery entrega")
    assert {r.trecho.secao for r in indice.buscar("delivery")} == {"Taxa", "Bebidas"}


@pytest.mark.parametrize(
    "pergunta, secao",
    [
        ("Qual o valor do frete?", "Área e taxa de entrega"),
        ("tem coca?", "Bebidas"),
        ("tem prato vegano?", "Opções vegetarianas e veganas"),
    ],
)
def test_sinonimos_do_restaurante(indice, pergunta, secao):
    """Sem os sinônimos, 'frete' trazia a seção de reembolso, e 'coca' não trazia nada."""
    resultados = indice.buscar(pergunta)
    assert resultados[0].trecho.secao == secao
    assert not Assistente(indice).responder(pergunta).recusa


def test_sinonimos_ilegiveis_tem_mensagem_clara(tmp_path, monkeypatch):
    pasta = sinonimos_em(tmp_path, "{}")

    def negar(self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(type(tmp_path), "read_bytes", negar)
    with pytest.raises(ErroDocumentos, match=r"não consegui ler sinonimos\.json: Permission denied"):
        carregar_sinonimos(pasta)


def test_numero_colado_na_unidade(indice):
    """ "5km" virava uma palavra só, que não existe nos documentos, e a relevância caía pela metade."""
    assert palavras("5km 23h30 R$9,00") == ["5", "km", "23", "h", "30", "r", "9", "00"]
    colado, separado = "Qual a taxa pra 5km?", "Qual a taxa pra 5 km?"
    assert indice.consulta(colado) == indice.consulta(separado)


@pytest.mark.parametrize("pergunta", ["vcs abrem que horas hj?", "q horas vc abre", "abre hj?", "oq tem de sobremesa"])
def test_abreviacoes_de_chat_nao_derrubam_a_pergunta(indice, pergunta):
    resposta = Assistente(indice).responder(pergunta)
    assert not resposta.recusa


@pytest.mark.parametrize(
    "pergunta, secao",
    [
        ("quanto custam os fretes?", "Área e taxa de entrega"),
        ("Tem cocas geladas?", "Bebidas"),
        ("tem refris?", "Bebidas"),
        ("que horas vocês abrem?", "Horário de funcionamento"),
        ("a cozinha fecha que horas?", "Horário de funcionamento"),
    ],
)
def test_sinonimo_vale_para_plural_e_variacoes(indice, pergunta, secao):
    """O sinônimo de "frete" não valia para "fretes": a pergunta era recusada."""
    assert indice.buscar(pergunta, 1)[0].trecho.secao == secao


def test_sinonimo_com_numero_e_recusado(tmp_path):
    with pytest.raises(ErroDocumentos, match="uma palavra só"):
        carregar_sinonimos(sinonimos_em(tmp_path, '{"5km": "cinco"}'))


MODELOS_DE_PERGUNTA = [
    "{s}",
    "como funciona {s}?",
    "qual a política de {s}?",
    "me fala sobre {s}",
    "oi, {s}?",
    "e {s}?",
    "quero saber de {s} hj",
]


def test_perguntar_pelo_titulo_da_secao_traz_a_secao(indice):
    """Um sinônimo mal escolhido ("funciona" -> "horário de funcionamento") mandava "como funciona
    o cancelamento?" para o horário. Este teste pega esse tipo de efeito colateral em todas as seções."""
    secoes = sorted({t.secao for t in indice.trechos})
    erradas = [
        (pergunta, resultados[0].trecho.secao if resultados else None)
        for secao in secoes
        for pergunta in [m.format(s=secao.lower()) for m in MODELOS_DE_PERGUNTA]
        if not (resultados := indice.buscar(pergunta, 1)) or resultados[0].trecho.secao != secao
    ]
    assert erradas == []


@pytest.mark.parametrize(
    "pergunta, secao",
    [
        ("como funciona o reembolso?", "Reembolso"),  # errava até sem sinônimos: "funciona" ~ "funcionamento"
        ("como funcionam os cupons?", "Cupons"),
        ("vocês funcionam domingo?", "Horário de funcionamento"),
        ("qual o horário de funcionamento?", "Horário de funcionamento"),
    ],
)
def test_funciona_e_verbo_generico(indice, pergunta, secao):
    assert indice.buscar(pergunta, 1)[0].trecho.secao == secao


@pytest.mark.parametrize("pergunta", ["a embalagem fecha bem?", "posso abrir uma reclamação?"])
def test_sinonimo_nao_puxa_pergunta_fora_do_escopo(indice, pergunta):
    """Os sinônimos "fecha" e "abrir" mandavam estas perguntas para o horário de funcionamento."""
    assert Assistente(indice).responder(pergunta).recusa


def test_sinonimos_nunca_estragam_uma_pergunta_do_gabarito(indice):
    """Toda pergunta dos dois gabaritos que acertava sem sinônimos continua acertando com eles."""
    sem = IndiceBM25(indice.trechos)
    for caminho in (GABARITO_PADRAO, GABARITO_COLOQUIAL):
        for caso in json.loads(caminho.read_text(encoding="utf-8"))["com_resposta"]:
            antes, depois = sem.buscar(caso["pergunta"], 1), indice.buscar(caso["pergunta"], 1)
            if antes and antes[0].trecho.secao == caso["secao"]:
                assert depois[0].trecho.secao == caso["secao"], caso["pergunta"]


@pytest.mark.parametrize(
    "md, esperado",
    [
        ("## Pizzas ##\ntexto", [("Pizzas", "texto")]),  # "#" de fechamento
        ("## C#\ntexto", [("C#", "texto")]),  # "#" colado no nome não é fechamento
        ("   ## Recuado\ntexto", [("Recuado", "texto")]),  # até 3 espaços ainda é título
        ("## A\n```\n## falso\n```\nfim", [("A", "```\n## falso\n```\nfim")]),  # dentro de código não é título
        ("## A\n~~~\n# falso\n~~~\nfim", [("A", "~~~\n# falso\n~~~\nfim")]),
        ("Pizzas\n======\ntexto um\n\nBebidas\n---\ntexto dois", [("Pizzas", "texto um"), ("Bebidas", "texto dois")]),
        ("## A\n- item\n---\nfim", [("A", "- item\n---\nfim")]),  # "---" depois de lista é separador
        ("## A\ntexto\n\n---\n\nmais", [("A", "texto\n\n---\n\nmais")]),  # "---" depois de linha vazia também
        ("## A\n```\ncódigo\n```\n---\nfim", [("A", "```\ncódigo\n```\n---\nfim")]),
        ("---\ntitle: Cardápio\ntags: [a]\n---\n## S\ncorpo", [("S", "corpo")]),  # metadados do topo
        ("---\nsem fim\n", [("x", "---\nsem fim")]),  # "---" sem fechamento não é bloco de metadados
        ("##\ntexto", [("x", "texto")]),  # título vazio usa o nome do arquivo
        ("    ## Código\ntexto", [("x", "## Código\ntexto")]),  # 4 espaços é bloco de código, não título
        ("## A\ntexto\n-\nmais", [("A", "texto\n-\nmais")]),  # um "-" sozinho não sublinha (pode ser item vazio)
        ("## A\n    código recuado\n===\nfim", [("A", "código recuado\n===\nfim")]),  # código recuado não é título
    ],
)
def test_markdown_comum(md, esperado):
    assert [(t.secao, t.texto) for t in dividir_markdown(md, "x.md")] == esperado


TIPOS_DE_LINHA = ["#", "##", "fechado", "recuado", "===", "---", "codigo", "texto", "texto", "lista", "vazia"]


def test_fuzz_markdown_com_codigo_sublinhado_e_metadados():
    """Mistura títulos (#, ##, com # no fim, recuados, sublinhados), blocos de código com
    falsos títulos dentro, listas, separadores e metadados. Cada linha de texto precisa
    aparecer uma vez, no trecho da seção certa, e nada de dentro do código vira seção."""
    gerador = random.Random(6)
    for _ in range(2000):
        linhas, esperado, secao = [], [], "x"
        if gerador.random() < 0.3:
            linhas += ["---", "title: metadado", "---"]
        for n in range(gerador.randint(1, 14)):
            tipo = gerador.choice(TIPOS_DE_LINHA)
            if tipo in ("#", "##"):
                secao = f"T{n}"
                linhas.append(f"{tipo} {secao}")
            elif tipo == "fechado":
                secao = f"T{n}"
                linhas.append(f"## {secao} ##")
            elif tipo == "recuado":
                secao = f"T{n}"
                linhas.append(f"{' ' * gerador.randint(1, 3)}## {secao}")
            elif tipo in ("===", "---"):
                secao = f"T{n}"
                linhas += ["", secao, tipo * 3]
            elif tipo == "codigo":
                cerca = gerador.choice(["```", "~~~"])
                linhas += [cerca, f"## falso{n}", f"codigo{n}", cerca]
                esperado += [(secao, f"codigo{n}")]
            elif tipo == "texto":
                linhas.append(f"linha{n}")
                esperado.append((secao, f"linha{n}"))
            elif tipo == "lista":
                linhas += [f"- item{n}", "---"]
                esperado.append((secao, f"item{n}"))
            else:
                linhas.append("")
        trechos = dividir_markdown("\n".join(linhas), "x.md")
        obtido = [(t.secao, palavra) for t in trechos for palavra in re.findall(r"(?:linha|item|codigo)\d+", t.texto)]
        assert obtido == esperado, "\n".join(linhas)
        assert not any(t.secao.startswith("falso") or "metadado" in t.texto for t in trechos), "\n".join(linhas)


def test_arquivos_ocultos_ficam_de_fora(tmp_path):
    """A trava do editor (".#entrega.md", link quebrado) derrubava o programa inteiro, e o
    "._cardapio.md" do macOS é lixo binário que virava trecho."""
    (tmp_path / "cardapio.md").write_text("## Bebidas\n\nSuco: R$ 12,00", encoding="utf-8")
    (tmp_path / "._cardapio.md").write_bytes(b"\x00\x05\x16\x07Mac OS X        \x00\x02ATTR\xff\xfe")
    try:
        (tmp_path / ".#cardapio.md").symlink_to(tmp_path / "nao-existe.md")
    except OSError:
        # No Windows, criar link simbólico exige administrador ou modo de desenvolvedor. O teste
        # falhou na primeira execução num Windows de verdade; sem o link, o resto continua valendo.
        (tmp_path / ".#cardapio.md").write_text("trava do editor", encoding="utf-8")
    (tmp_path / "pasta.md").mkdir()
    assert [(t.fonte, t.secao) for t in carregar_pasta(tmp_path)] == [("cardapio.md", "Bebidas")]


def test_lista_longa_sem_linha_vazia_e_quebrada_por_item():
    """Um cardápio colado sem linhas vazias virava um trecho de 519 palavras."""
    itens = [f"- Sabor {n}: molho, muçarela e mais quatro ingredientes da casa. R$ {50 + n},00" for n in range(45)]
    trechos = dividir_markdown("## Pizzas\n" + "\n".join(itens), "c.md")
    assert len(trechos) > 1 and all(len(t.texto.split()) <= 120 for t in trechos)
    assert [linha for t in trechos for linha in t.texto.splitlines()] == itens  # nenhum item perdido ou partido
    assert {t.secao for t in trechos} == {"Pizzas"}


def test_paragrafo_longo_de_uma_linha_so_fica_inteiro():
    texto = " ".join(["palavra"] * 300)
    assert [t.texto for t in dividir_markdown(f"## A\n{texto}", "x.md")] == [texto]


def test_itens_repetidos_e_paragrafos_mantem_o_separador_certo():
    repetidos = "\n".join(["- Item igual com várias palavras para encher a linha toda aqui"] * 30)
    md = f"## A\nIntrodução curta.\n\n{repetidos}\n\nFecho curto."
    texto = "\n".join(t.texto for t in dividir_markdown(md, "x.md"))
    assert "Introdução curta.\n\n- Item" in texto and "aqui\n- Item" in texto
    assert "aqui\n\n- Item" not in texto and texto.endswith("aqui\n\nFecho curto.")
