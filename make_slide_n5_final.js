const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";

const FONT = "Malgun Gothic";
const INK = "1a1a1a", SUB = "5a5955", MUT = "8a8880", GOOD = "2e7d32", WARN = "b71c1c";

const s = pres.addSlide();
s.background = { color: "FFFFFF" };

s.addText("국면 라벨 최종 결정: N5 채택", {
  x: 0.55, y: 0.3, w: 12.2, h: 0.55, fontFace: FONT, fontSize: 26, bold: true, color: INK, margin: 0,
});
s.addText("사전 등록 7시도(N1~N8·J1) · 채택 발효 = 팀 머지 · 결정문 docs/REGIME_FINAL_DECISION.md · 실험 전문 regime_next_실험로그.md", {
  x: 0.55, y: 0.88, w: 12.2, h: 0.32, fontFace: FONT, fontSize: 11.5, color: SUB, margin: 0,
});

// ---- 좌: 무엇이 바뀌나 + 왜 N7이 아닌가 ----
s.addText("무엇이 바뀌나 — 물가 축 하나 (튜닝 파라미터 0개)", {
  x: 0.55, y: 1.42, w: 5.9, h: 0.3, fontFace: FONT, fontSize: 13.5, bold: true, color: INK, margin: 0,
});
s.addText([
  { text: "창 내 중앙값 → 당시 한국은행 물가안정목표 (창-독립)", options: { bold: true, fontSize: 12, color: INK, breakLine: true } },
  { text: "상향은 목표 초과 즉시, 하향은 목표 −0.5%p(설명책임 폭) 미만일 때만. 창 불일치의 79%가 물가축(저인플레기 집중)이라는 진단의 직접 처방. 성장 축·분기 구조·워크포워드·2026 홀드아웃은 V2 그대로.", options: { fontSize: 11, color: SUB, breakLine: true, paraSpaceAfter: 12 } },
  { text: "왜 전 기준 합격한 N7이 아니라 N5인가", options: { bold: true, fontSize: 12, color: INK, breakLine: true } },
  { text: "확률 레이어 실측이 결정타 — N7 라벨로 학습하면 나우캐스트 스태그 확률이 2024년 78%로 경화(완화 신호 소거), N5는 56%로 하락하며 2024년 완화를 표현. 화면 판정의 정직성 > 라벨의 매끈함.", options: { fontSize: 11, color: SUB } },
], { x: 0.55, y: 1.78, w: 5.9, h: 2.7, fontFace: FONT, valign: "top", margin: 0, lineSpacingMultiple: 1.2 });

s.addText("기준 미달 1건 — 명기 채택 (완화 아님)", {
  x: 0.55, y: 4.55, w: 5.9, h: 0.3, fontFace: FONT, fontSize: 13.5, bold: true, color: INK, margin: 0,
});
s.addText([
  { text: "3개월 이하 구간 24개 (사전 등록 기준 ≤23) — 후처리 없이 유지.", options: { fontSize: 11.5, color: SUB, breakLine: true } },
  { text: "사후 병합은 미래 정보 사용(비인과)이며, A–B–A 후보 17개에 2021-09·2022-03 위기 구간이 포함돼 균일 병합은 앵커를 파괴. 기준을 지킨 채 미달을 공개하는 것이 검증 문화(T1 기각·음수 초과성과 공개)와 같은 문법.", options: { fontSize: 11, color: MUT } },
], { x: 0.55, y: 4.9, w: 5.9, h: 1.6, fontFace: FONT, valign: "top", margin: 0, lineSpacingMultiple: 1.2 });

// ---- 우: 성적표 ----
const X = 6.9, W = 5.9;
s.addText("채택 성적 (V2 → N5)", {
  x: X, y: 1.42, w: W, h: 0.3, fontFace: FONT, fontSize: 13.5, bold: true, color: INK, margin: 0,
});
const rows = [
  ["", "V2", "N5", ""],
  ["창 민감도 (10y vs 5y)", "68.3%", "93.3%", "+25.0%p"],
  ["3창 전체 일치", "63.3%", "92.2%", "+28.9%p"],
  ["위기 앵커 (6종)", "6/6", "6/6", "유지"],
  ["전환 / 최장 구간", "33회/24개월", "33회/24개월", "민감도 보존"],
  ["3개월 이하 구간", "23개", "24개 (+1)", "미달 명기"],
  ["2024 스태그 확률(연평균)", "—", "56%", "완화 표현"],
  ["ERC 4국면 비중", "기준", "차이 ≤1.2%p", "배분 안정"],
];
const tY = 1.8, rH = 0.42;
rows.forEach((r, i) => {
  const y = tY + i * rH;
  if (i > 0 && i % 2 === 1) {
    s.addShape("rect", { x: X - 0.06, y, w: W + 0.12, h: rH, fill: { color: "F6F6F4" }, line: { type: "none" } });
  }
  const base = i === 0 ? { bold: true, color: MUT, fontSize: 11 } : { color: SUB, fontSize: 11 };
  s.addText(r[0], { x: X + 0.04, y, w: 2.6, h: rH, fontFace: FONT, valign: "middle", margin: 0, ...base, color: i === 0 ? MUT : INK, bold: i === 0 });
  s.addText(r[1], { x: X + 2.7, y, w: 1.2, h: rH, fontFace: FONT, valign: "middle", margin: 0, ...base });
  s.addText(r[2], { x: X + 3.9, y, w: 1.25, h: rH, fontFace: FONT, valign: "middle", margin: 0, ...base, bold: true, color: i === 0 ? MUT : INK });
  s.addText(r[3], { x: X + 5.1, y, w: 0.8, h: rH, fontFace: FONT, valign: "middle", margin: 0, fontSize: 10, color: i === 0 ? MUT : (r[3] === "미달 명기" ? WARN : GOOD), bold: i > 0 });
});

s.addText("채택 이후 완료된 후속 조치", {
  x: X, y: 5.35, w: W, h: 0.3, fontFace: FONT, fontSize: 13.5, bold: true, color: INK, margin: 0,
});
s.addText("잠정 판정 엔진(라벨 공백기 화면 개선, 채점 합격) · 운영 판정 월간 로거 · 사후 확정 라벨 2-트랙(NBER 방식) · HMM 벤치마크(위기 전환 수렴 확인) · 논문 Discussion 초안", {
  x: X, y: 5.7, w: W, h: 0.9, fontFace: FONT, fontSize: 11, color: SUB, margin: 0, lineSpacingMultiple: 1.25,
});

s.addText("규칙: 물가축 상향 CPI>T(t) 즉시 · 하향 CPI<T(t)−0.5%p (T: 2016~ 2.0 / 이전 3.0) · 성장축 GDP QoQ vs 10y 창 중앙값(V2 동일) · 재현 scripts/regime_next_n5.py",
  { x: 0.55, y: 6.95, w: 12.2, h: 0.3, fontFace: FONT, fontSize: 10, color: MUT, margin: 0 });

pres.writeFile({ fileName: "n5_final_decision.pptx" }).then(() => console.log("done"));
