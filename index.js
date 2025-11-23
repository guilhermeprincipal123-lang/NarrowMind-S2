const fs = require('fs');

const data = fs.readFileSync("./input.txt").toString();

function parseTokens() {
    const tokens = data.trim().split(/[^\p{L}\p{N}]+/u).filter(Boolean);
    return tokens;
}

