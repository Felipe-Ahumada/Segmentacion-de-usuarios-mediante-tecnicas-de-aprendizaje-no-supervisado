CREATE TABLE perfil_usuario
(
    id_cliente INT PRIMARY KEY,
    edad INT,
    dispositivos_registrados INT,
    porcentaje_uso_app_movil NUMERIC,
    cantidad_perfiles_creados INT,
    interacciones_mensuales_soporte INT,
    distancia_promedio_red_km NUMERIC
);

COPY perfil_usuario
FROM '/docker-entrypoint-initdb.d/perfil_usuarios.csv'
DELIMITER ','
CSV HEADER;
