#!/bin/bash

# Кольори для виводу
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Запуск Kubernetes Autoscaler...${NC}"

# Перевірка Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker не встановлено!${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose не встановлено!${NC}"
    exit 1
fi

# Запуск
echo -e "${GREEN}📦 Збірка та запуск контейнерів...${NC}"
docker compose up --build -d

# Перевірка статусу
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Всі сервіси запущені успішно!${NC}"
    echo ""
    echo -e "${YELLOW}📊 Доступні сервіси:${NC}"
    echo -e "   • ${GREEN}Grafana (Моніторинг):${NC} http://localhost:3000 (admin/admin)"
    echo -e "   • ${GREEN}Dashboard:${NC} http://localhost:8080"
    echo -e "   • ${GREEN}API:${NC} http://localhost:5000"
    echo -e "   • ${GREEN}Loki:${NC} http://localhost:3100"
    echo ""
    echo -e "${YELLOW}📝 Корисні команди:${NC}"
    echo -e "   • Переглянути логи: ${GREEN}docker-compose logs -f${NC}"
    echo -e "   • Статус сервісів: ${GREEN}docker-compose ps${NC}"
    echo -e "   • Зупинити: ${GREEN}./end.sh${NC}"
    echo ""
    echo -e "${YELLOW}💡 Порада:${NC} Відкрийте Grafana для real-time моніторингу!"
else
    echo -e "${RED}❌ Помилка при запуску!${NC}"
    exit 1
fi