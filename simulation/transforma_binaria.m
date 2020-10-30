function s=transforma_binaria(serie,num_simbolos,umbral)

% Funcion:   transformar_senal
%
%                   Transforma una señal temporal en una serie compuesta por un
%                   número de símbolos determinado
%
% Argumentos de entrada:
%   serie:          señal temporal a transformar
%   num_simbolos:   número de símbolos a utilizar en la transformación (2 ó 3)
%   umbral:         umbral de decisión a aplicar en la transformación
% Argumentos de salida:
%   s:              señal transformada
%
% Ultima modificacion:   11 de Diciembre de 2007
%
% Autor:            Alicia Rodrigo de Diego y Jose Victor Marcos Martin
% Modificado por:   Daniel Álvarez González
%

    if num_simbolos==2
        if strcmp(umbral,'mediana')  
            mediana=median(serie);
            for i=1:1:length(serie)
                if serie(i)<mediana
                    s(i)=0;
                else
                    s(i)=1;
                end
            end
        else
            media=mean(serie);
            for i=1:1:length(serie)
                if serie(i)<media
                    s(i)=0;
                else
                    s(i)=1;
                end
            end
        end
        
    else
        mediana=median(serie);
        mx=abs(max(serie));
        mn=abs(min(serie));
        
        td1=mediana-mn/16;
        td2=mediana+mx/16;
        
        for i=1:1:length(serie)
            if serie(i)<=td1
                s(i)=0;
            else if serie(i)<td2
                    s(i)=1;
                else
                    s(i)=2;
                end
            end
        end
    end