document.getElementById("form-analise").addEventListener("submit", async function(event){
    event.preventDefault();


const textoDigitado = document.getElementById("texto-de-avaliacao").value;
const botao = document.getElementById("btn-analisar");
const resultadoBox = document.getElementById("resultado-box");

botao.disabled = true;
botao.innerText = "Processando...";
resultadoBox.style.display = "none";

try{
    const resposta = await fetch("/api/analisar",{
        method : "POST",
        headers : { "Content-Type" : "application/json" }, 
        body : JSON.stringify({avaliacao: textoDigitado})
    });

    const dados = await resposta.json();

    if(resposta.ok){
        document.getElementById("texto-exibido").innerHTML = '""' + dados.texto_analisado + '""';
        
        const spanVeredito = document.getElementById("veredito-exibido");
        spanVeredito.innerText = dados.resultado + (dados.resultado === "Positivo" ? "✅" : "⚠️");
        spanVeredito.className = dados.resultado;

        document.getElementById("confianca-exibido").innerText = dados.confianca;
        
        resultadoBox.className = 'borda-' + dados.resultado;
        
        resultadoBox.style.display = 'block';
        
    } else{
        alert("Erro na IA: " + dados.erro);
    }
 
    }catch (erro) {
        alert("Erro de conexão com o servidor Python.");
    }finally{
        botao.disabled = false;
        botao.innerText = "Classificar Sentimentos";
    }

});