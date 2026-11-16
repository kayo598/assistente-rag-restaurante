import json

import pytest

from rag_restaurante.assistente import Assistente
from rag_restaurante.avaliacao import GABARITO_COLOQUIAL, GABARITO_PADRAO, ErroGabarito, avaliar
from rag_restaurante.llm import ErroLLM

from .conftest import ClienteFixo


def test_avaliacao_offline_mantem_qualidade_minima(indice):
    """Protege contra regressão na busca e na resposta."""
    m = avaliar(indice, GABARITO_PADRAO, Assistente(indice))
    assert m.total == 23
    assert m.acerto_1 >= 0.9 and m.acerto_3 == 1.0
    assert m.resposta_correta >= 0.9 and m.cita_fonte_certa >= 0.9


def test_resposta_errada_conta_como_erro(indice):
    """Um assistente que acha o trecho mas responde errado precisa aparecer na avaliação."""
    m = avaliar(indice, GABARITO_PADRAO, Assistente(indice, ClienteFixo("Resposta qualquer [1].")))
    assert m.acerto_1 >= 0.9  # a busca continua boa...
    assert m.resposta_correta == 0  # ...mas nenhuma resposta tem o fato esperado
    assert m.recusa_correta == 0.5  # 2 das 4 nem chegam ao LLM (busca sem resultado)


def gabarito(tmp_path, dados):
    arquivo = tmp_path / "g.json"
    arquivo.write_text(dados if isinstance(dados, str) else json.dumps(dados), encoding="utf-8")
    return arquivo


CASO = {"pergunta": "Abre?", "fonte": "entrega.md", "secao": "Horário de funcionamento", "deve_conter": ["segunda"]}


@pytest.mark.parametrize(
    "dados, mensagem",
    [
        ("{x", "não é um JSON válido"),
        ({"sem_resposta": []}, "com_resposta"),
        ({"com_resposta": [{**CASO, "secao": "Horario"}], "sem_resposta": []}, "não existe nos documentos"),
        ({"com_resposta": [{**CASO, "deve_conter": []}], "sem_resposta": []}, "lista de textos"),
        ({"com_resposta": [{"pergunta": "x"}], "sem_resposta": []}, "faltam os campos"),
    ],
)
def test_gabarito_invalido(indice, tmp_path, dados, mensagem):
    with pytest.raises(ErroGabarito, match=mensagem):
        avaliar(indice, gabarito(tmp_path, dados), Assistente(indice))


def test_gabarito_inexistente(indice, tmp_path):
    with pytest.raises(ErroGabarito, match="não consegui ler"):
        avaliar(indice, tmp_path / "nao.json", Assistente(indice))


def test_fatos_do_gabarito_existem_nos_documentos(indice):
    """O próprio gabarito pode ter erro: cada fato esperado precisa estar na seção indicada."""
    dados = json.loads(GABARITO_PADRAO.read_text(encoding="utf-8"))
    for caso in dados["com_resposta"]:
        texto = " ".join(
            t.texto for t in indice.trechos if t.fonte == caso["fonte"] and t.secao == caso["secao"]
        ).casefold()
        for fato in caso["deve_conter"]:
            assert fato.casefold() in texto, (caso["pergunta"], fato)


@pytest.mark.parametrize(
    "dados, mensagem",
    [
        # string em vez de lista: era conferida letra por letra, e "R$ 9,00" passava com "R$ 90,00"
        ({"com_resposta": [{**CASO, "deve_conter": "R$ 9,00"}], "sem_resposta": []}, "lista de textos"),
        ({"com_resposta": [{**CASO, "deve_conter": "segunda"}], "sem_resposta": []}, "lista de textos"),  # sem espaço
        ({"com_resposta": [{**CASO, "deve_conter": [9]}], "sem_resposta": []}, "lista de textos"),
        ({"com_resposta": [{**CASO, "deve_conter": ["ok", ""]}], "sem_resposta": []}, "lista de textos"),
        ({"com_resposta": [{**CASO, "pergunta": 123}], "sem_resposta": []}, "texto não vazio"),  # era traceback
        ({"com_resposta": [{**CASO, "pergunta": "x" * 1001}], "sem_resposta": []}, "passa de 1000"),
        ({"com_resposta": ["Abre?"], "sem_resposta": []}, "precisa ser um objeto"),
        ({"com_resposta": [CASO], "sem_resposta": "Tem wifi?"}, "precisa das listas"),  # virava uma pergunta por letra
        ({"com_resposta": [CASO], "sem_resposta": "wifi"}, "precisa das listas"),
        ({"com_resposta": [CASO], "sem_resposta": [None]}, "sem_resposta 1"),
        ({"com_resposta": {"a": CASO}, "sem_resposta": []}, "com_resposta"),
    ],
)
def test_gabarito_com_tipos_errados(indice, tmp_path, dados, mensagem):
    with pytest.raises(ErroGabarito, match=mensagem):
        avaliar(indice, gabarito(tmp_path, dados), Assistente(indice))


class ClienteInstavel:
    """Falha com erro temporário nas chamadas indicadas (contando de 1)."""

    def __init__(self, falhas, fatal=False):
        self.falhas, self.fatal, self.chamadas = set(falhas), fatal, 0

    def completar(self, sistema, mensagem):
        self.chamadas += 1
        if self.chamadas in self.falhas:
            raise ErroLLM("API respondeu 529; desisti após 4 tentativas", fatal=self.fatal)
        return "Resposta [1]."


def test_falha_isolada_da_api_nao_perde_a_avaliacao(indice):
    """Antes, uma falha temporária na 5ª pergunta jogava fora as 4 respostas já pagas."""
    cliente = ClienteInstavel({5, 9})
    m = avaliar(indice, GABARITO_PADRAO, Assistente(indice, cliente))
    assert m.falhas_api == 2 and m.total == 23
    assert sum(e.startswith("api ") for e in m.erros) == 2
    assert m.cita_fonte_certa < 1  # as perguntas que falharam contam como erro


def test_falha_na_parte_fora_do_escopo_conta_como_erro(indice):
    """As 23 perguntas com resposta chamam o LLM; a 24ª chamada é a 1ª pergunta fora do escopo."""

    class Recusa(ClienteInstavel):
        def completar(self, sistema, mensagem):
            super().completar(sistema, mensagem)
            return "Não encontrei essa informação."

    assert avaliar(indice, GABARITO_PADRAO, Assistente(indice, Recusa(set()))).recusa_correta == 1.0
    m = avaliar(indice, GABARITO_PADRAO, Assistente(indice, Recusa({24})))
    assert m.falhas_api == 1 and m.recusa_correta == 0.75


def test_varias_falhas_seguidas_param_a_avaliacao(indice):
    cliente = ClienteInstavel({3, 4, 5, 6})
    with pytest.raises(ErroLLM, match="3 perguntas seguidas falharam"):
        avaliar(indice, GABARITO_PADRAO, Assistente(indice, cliente))
    assert cliente.chamadas == 5  # parou na 3ª falha seguida, sem gastar o resto


def test_falhas_alternadas_nao_param(indice):
    m = avaliar(indice, GABARITO_PADRAO, Assistente(indice, ClienteInstavel({2, 3, 5, 6, 8, 9})))
    assert m.falhas_api == 6


def test_erro_fatal_para_na_hora(indice):
    cliente = ClienteInstavel({2}, fatal=True)
    with pytest.raises(ErroLLM) as erro:
        avaliar(indice, GABARITO_PADRAO, Assistente(indice, cliente))
    assert erro.value.fatal and cliente.chamadas == 2


def test_conjunto_coloquial_mantem_qualidade_minima(indice):
    """Conjunto separado, nunca usado para escolher sinônimos: na criação, acertava 72% das respostas."""
    m = avaliar(indice, GABARITO_COLOQUIAL, Assistente(indice))
    assert m.total == 18 and m.resposta_correta >= 0.8 and m.recusa_correta == 1.0


def test_busca_da_avaliacao_e_a_mesma_do_assistente(indice, tmp_path):
    """A avaliação media a busca com a saudação; o assistente busca sem ela. Com a saudação,
    "bom dia, quais pizzas vocês têm?" ia para o reembolso ("dia" casa com "1 dia útil")."""
    caso = {
        "pergunta": "bom dia, quais pizzas vocês têm?",
        "fonte": "cardapio.md",
        "secao": "Pizzas",
        "deve_conter": ["Margherita"],
    }
    assert indice.buscar(caso["pergunta"], 1)[0].trecho.secao == "Reembolso"  # a busca crua erra
    m = avaliar(indice, gabarito(tmp_path, {"com_resposta": [caso], "sem_resposta": []}), Assistente(indice))
    assert m.acerto_1 == 1.0 and m.recusa_correta is None
