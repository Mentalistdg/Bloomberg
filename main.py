"""Prueba de conexión a Bloomberg vía blpapi (Desktop API / DAPI).

Requisitos:
- Bloomberg Terminal abierto y logueado.
- bbcomm.exe corriendo (escucha localhost:8194).
- Paquete blpapi instalado en el venv.
"""

import blpapi


def main() -> None:
    options = blpapi.SessionOptions()
    options.setServerHost("localhost")
    options.setServerPort(8194)

    session = blpapi.Session(options)
    if not session.start():
        raise RuntimeError("No se pudo iniciar la sesión con bbcomm (¿Terminal abierta?)")
    if not session.openService("//blp/refdata"):
        raise RuntimeError("No se pudo abrir //blp/refdata")

    refdata = session.getService("//blp/refdata")
    request = refdata.createRequest("ReferenceDataRequest")
    request.append("securities", "IBM US Equity")
    request.append("securities", "AAPL US Equity")
    request.append("fields", "PX_LAST")
    request.append("fields", "NAME")

    session.sendRequest(request)

    while True:
        event = session.nextEvent(5000)
        for msg in event:
            print(msg)
        if event.eventType() == blpapi.Event.RESPONSE:
            break

    session.stop()


if __name__ == "__main__":
    main()
