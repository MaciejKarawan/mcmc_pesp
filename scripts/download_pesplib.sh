#!/bin/bash
# Skrypt pobierania instancji PESPlib
# Użycie: ./scripts/download_pesplib.sh

set -e

PESPLIB_BASE_URL="https://num.math.uni-goettingen.de/~m.goerigk/pesplib"
OUTPUT_DIR="data/pesplib/raw"

echo "=== Pobieranie instancji PESPlib ==="

# Utwórz katalog
mkdir -p "$OUTPUT_DIR"

# Funkcja pobierania
download_instance() {
    local name=$1
    local url="${PESPLIB_BASE_URL}/instances/${name}"
    
    echo "Pobieranie: $name"
    
    if command -v wget &> /dev/null; then
        wget -q -P "$OUTPUT_DIR" "$url" 2>/dev/null || echo "  [WARN] Nie udało się pobrać $name"
    elif command -v curl &> /dev/null; then
        curl -s -o "$OUTPUT_DIR/$(basename $url)" "$url" 2>/dev/null || echo "  [WARN] Nie udało się pobrać $name"
    else
        echo "  [ERROR] Brak wget ani curl"
        exit 1
    fi
}

# Pobierz główną stronę z listą instancji
echo "Sprawdzanie dostępnych instancji..."

# Lista znanych instancji (backup jeśli strona niedostępna)
KNOWN_INSTANCES=(
    "toy"
    "grid-1"
    "grid-2" 
    "grid-3"
    "simplex-1"
    "simplex-2"
)

echo ""
echo "Pobieranie instancji..."

for inst in "${KNOWN_INSTANCES[@]}"; do
    # Pobierz pliki instancji
    download_instance "${inst}/Events.giv"
    download_instance "${inst}/Activities.giv"
    download_instance "${inst}/Config.cnf"
    download_instance "${inst}/${inst}.opt" 2>/dev/null || true
    download_instance "${inst}/${inst}.tim" 2>/dev/null || true
done

echo ""
echo "=== Pobieranie zakończone ==="
echo "Instancje zapisane w: $OUTPUT_DIR"
echo ""
echo "Następny krok: python -m src.io.pesplib_loader $OUTPUT_DIR data/pesplib/converted/"
