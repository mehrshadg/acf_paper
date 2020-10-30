function FrecuenciaMediana = CalculoMF(PSD, freq, banda)
% CALCULOMF Calcula la frecuencia mediana de la distribucion
%       contenida en PSD.
%
%		FRECUENCIAMEDIANA = CALCULOMF(PSD, F, BANDA), calcula la frecuencia 
%       mediana de la distribucion PSD, indexada por el vector F, entre las 
%       frecuencias indicadas en BANDA.
%
%       En FRECUENCIAMEDIANA se devuelve la frecuencia mediana calculada.
%
% See also CALCULARPARAMETRO, CALCULOSEF, CALCULOIAFTF

%
% Versión: 2.0
%
% Fecha de creación: 13 de Junio de 2005
%
% Última modificación: 25 de ctubre de 2010
%
% Autor: Jesús Poza Crespo
% Modificado: Javier Gomez-Pilar 17/05/2017
%

% Se inicializa la variable de salida.
FrecuenciaMediana = [];

% Se buscan los índices positivos en la banda de paso
indbanda = min(find(freq >= banda(1))) : max(find(freq <= banda(2)));


% Potencia total para el espectro positivo
potenciatotal = sum(PSD(indbanda));
% Se suman los valores de potencia relativa para el espectro positivo
vectorsuma = cumsum(PSD(indbanda));

% Se coge el índice para el cual se tiene la mitad de la potencia total.
indmitad = max(find(vectorsuma <= (potenciatotal/2)));
indmedia = indbanda(indmitad);

% Se toma la frecuencia con la potencia media (frecuencia mediana)
% En caso de que la PSD no esté definida, la MF tampoco
if isnan(PSD(indbanda(1))),
    FrecuenciaMediana = NaN;
% Si no se ha seleccionado ningún índice es porque en el primer valor esta
% mas del 50% de la potencia total
else
    if isempty(indmedia),
        indmedia = indbanda(1);
        FrecuenciaMediana = freq(indmedia);
    else
        FrecuenciaMediana = freq(indmedia);
    end
end % Fin del 'if' que comprueba si hay algun índice

clear PSD f banda indbanda potencia toal vectorsuma indmitad indmedia


