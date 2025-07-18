import qrcode
from rich.console import Console


def show_qr_ascii(code: str):
    qr = qrcode.QRCode(version=1, box_size=1, border=0)
    qr.add_data(code)
    qr.make()
    matrix = qr.get_matrix()
    console = Console()
    console.print("\nScan this QR with your WhatsApp:\n")
    for y in range(0, len(matrix), 2):
        row = ""
        for x in range(len(matrix[0])):
            top = matrix[y][x]
            bottom = matrix[y + 1][x] if y + 1 < len(matrix) else 0
            row += "█" if top and bottom else "▀" if top else "▄" if bottom else " "
        console.print(row)
