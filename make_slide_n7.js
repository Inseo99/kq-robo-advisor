const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";

const FONT = "Malgun Gothic";
const INK = "1a1a1a", SUB = "5a5955", MUT = "8a8880", GOOD = "2e7d32";

const s = pres.addSlide();
s.background = { color: "FFFFFF" };

s.addText("국면 라벨 차기 버전(N7) 채택 심의 요약", {
  x: 0.55, y: 0.3, w: 12.2, h: 0.55, fontFace: FONT, fontSize: 26, bold: true, color: INK, margin: 0,
});
s.addText("regime-next 브랜치 · 사전 등록 5시도(기각 1·대체 1·대안 2·합격 1) · 채택 라벨·홀드아웃·발표 수치 불가침 유지 · 상세: docs/REGIME_N7_ADOPTION.md", {
  x: 0.55, y: 0.88, w: 12.2, h: 0.32, fontFace: FONT, fontSize: 11.5, color: SUB, margin: 0,
});

// ---- 좌: 변경점 2개 ----
s.addText("무엇이 바뀌나 (규칙 2개, 튜닝 파라미터 0개)", {
  x: 0.55, y: 1.42, w: 5.9, h: 0.3, fontFace: FONT, fontSize: 13.5, bold: true, color: INK, margin: 0,
});
s.addText([
  { text: "물가 축 — 창 내 중앙값 → 당시 물가안정목표(창-독립)", options: { bold: true, fontSize: 12, color: INK, breakLine: true } },
  { text: "상향은 목표 초과 즉시, 하향은 목표 −0.5%p(한은 설명책임 폭) 미만일 때만. 창 불일치의 79%가 물가축이라는 진단에 대한 직접 처방.", options: { fontSize: 11, color: SUB, breakLine: true, paraSpaceAfter: 10 } },
  { text: "성장 축 — 위험 해제만 강도 조건부 (진입은 V2와 동일·즉시)", options: { bold: true, fontSize: 12, color: INK, breakLine: true } },
  { text: "중앙값 상회 폭이 75분위 이상이면 즉시 인정, 미만이면 2분기 확인. 기각됐던 양방향 k-확인과 달리 위기 진입 무지연 — 위기 앵커 6/6 보존.", options: { fontSize: 11, color: SUB } },
], { x: 0.55, y: 1.78, w: 5.9, h: 2.5, fontFace: FONT, valign: "top", margin: 0, lineSpacingMultiple: 1.2 });

// ---- 좌하: 2026 홀드아웃 ----
s.addText("2026 홀드아웃 대조 (실운영 판정과 수기 대조용)", {
  x: 0.55, y: 4.05, w: 5.9, h: 0.3, fontFace: FONT, fontSize: 13.5, bold: true, color: INK, margin: 0,
});
s.addText([
  { text: "01~02월 스태그플레이션 (GDP −0.1 · CPI 2.0)", options: { fontSize: 11.5, color: SUB, breakLine: true } },
  { text: "03~06월 리플레이션 (GDP +1.8 반등 · CPI 2.2→3.2)", options: { fontSize: 11.5, color: SUB, breakLine: true } },
  { text: "V2·N7 전 구간 동일 판정, N7은 3창 만장일치 — 검증 질문은 \"실운영이 3월 전환을 얼마나 빨리 따라갔나\"", options: { fontSize: 11, color: MUT } },
], { x: 0.55, y: 4.4, w: 5.9, h: 1.4, fontFace: FONT, valign: "top", margin: 0, lineSpacingMultiple: 1.25 });

// ---- 우: 성적표 표 ----
const X = 6.9, W = 5.9;
s.addText("사전 등록 채점 (전 기준 합격 5/5)", {
  x: X, y: 1.42, w: W, h: 0.3, fontFace: FONT, fontSize: 13.5, bold: true, color: INK, margin: 0,
});
const rows = [
  ["", "V2 현행", "N7", ""],
  ["창 민감도 (10y vs 5y)", "68.3%", "91.7%", "+23.4%p"],
  ["3창 전체 일치", "63.3%", "85.0%", "+21.7%p"],
  ["위기 앵커 (6종)", "6/6", "6/6", "유지"],
  ["전환 횟수 (15년)", "33회", "19회", "적정 빈도"],
  ["3개월 이하 구간", "23개", "12개", "−11개"],
  ["2022 스태그 진입", "10y만 3월", "3창 모두 3월", "동기화"],
  ["ERC 4국면 비중", "—", "차이 ≤0.3%p", "배분 안정"],
];
const tY = 1.8, rH = 0.44;
rows.forEach((r, i) => {
  const y = tY + i * rH;
  if (i > 0 && i % 2 === 1) {
    s.addShape("rect", { x: X - 0.06, y, w: W + 0.12, h: rH, fill: { color: "F6F6F4" }, line: { type: "none" } });
  }
  const opts = i === 0 ? { bold: true, color: MUT, fontSize: 11 } : { color: SUB, fontSize: 11.5 };
  s.addText(r[0], { x: X + 0.04, y, w: 2.5, h: rH, fontFace: FONT, valign: "middle", margin: 0, ...opts, color: i === 0 ? MUT : INK, bold: i === 0 });
  s.addText(r[1], { x: X + 2.6, y, w: 1.1, h: rH, fontFace: FONT, valign: "middle", margin: 0, ...opts });
  s.addText(r[2], { x: X + 3.75, y, w: 1.3, h: rH, fontFace: FONT, valign: "middle", margin: 0, ...opts, bold: true, color: i === 0 ? MUT : INK });
  s.addText(r[3], { x: X + 5.05, y, w: 0.85, h: rH, fontFace: FONT, valign: "middle", margin: 0, fontSize: 10.5, color: i === 0 ? MUT : GOOD, bold: i > 0 });
});

// ---- 우하: 남은 절차 ----
s.addText("머지 전 확인 절차", {
  x: X, y: 5.5, w: W, h: 0.3, fontFace: FONT, fontSize: 13.5, bold: true, color: INK, margin: 0,
});
s.addText("패치 적용·푸시  →  QA §8 가격 검증  →  서버 재시작+Ctrl+F5 화면 확인  →  팀 머지(= 공식 채택)", {
  x: X, y: 5.85, w: W, h: 0.6, fontFace: FONT, fontSize: 11.5, color: SUB, margin: 0, lineSpacingMultiple: 1.25,
});

s.addText("실험 전문: regime_next_실험로그.md · 재현: scripts/regime_next_n7.py · 대안 보존: N5(민감도 최우선)·N6(안정성 최우선) · 구본 백업: labels_v2_adopted_backup.csv",
  { x: 0.55, y: 6.95, w: 12.2, h: 0.3, fontFace: FONT, fontSize: 10, color: MUT, margin: 0 });

pres.writeFile({ fileName: "n7_adoption_review.pptx" }).then(() => console.log("done"));
