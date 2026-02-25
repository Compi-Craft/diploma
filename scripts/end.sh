#!/bin/bash

# Кольори для виводу
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🛑 Зупинка Kubernetes Autoscaler...${NC}"

# Показати які контейнери працюють
echo -e "${YELLOW}📋 Запущені контейнери:${NC}"
docker-compose ps

echo ""
read -p "Зупинити всі сервіси та видалити volumes? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}⏳ Зупинка та видалення...${NC}"
    docker compose down -v
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Всі сервіси зупинені та volumes видалені${NC}"
    else
        echo -e "${RED}❌ Помилка при зупинці!${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⏳ Зупинка без видалення volumes...${NC}"
    docker compose down
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Всі сервіси зупинені (volumes збережено)${NC}"
    else
        echo -e "${RED}❌ Помилка при зупинці!${NC}"
        exit 1
    fi
fi 