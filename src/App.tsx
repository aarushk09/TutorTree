import type { CSSProperties, FormEvent } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Scenario } from "./data/humanRatingTask";
import { scenarios } from "./data/humanRatingTask";
import { isSupabaseConfigured, supabase } from "./supabaseClient";

type Selection = "A" | "B" | "Tie";
type Stage =
  | "landing"
  | "tour"
  | "prestart"
  | "evaluation"
  | "feedback"
  | "finished";
type WarningLabel = "answered_under_3_seconds";
type TourFocus = "context" | "interventions" | "tie" | "reasoning";
type EvaluatorRole =
  | "University Faculty"
  | "K-12 Teacher"
  | "Student"
  | "EdTech Researcher"
  | "Other";

const SESSION_KEY = "neurips-human-eval-session-id";
const EVALUATOR_ROLE_KEY = "neurips-human-eval-evaluator-role";
const QUALITY_REMINDER_KEY = "neurips-human-eval-hide-quality-reminder";
const FAST_RESPONSE_THRESHOLD_MS = 3000;
const QUALITY_REMINDER_SECONDS = 5;
const GUIDE_GAP = 22;
const GUIDE_MARGIN = 14;
const evaluatorRoles: EvaluatorRole[] = [
  "University Faculty",
  "K-12 Teacher",
  "Student",
  "EdTech Researcher",
  "Other",
];

const tourSteps: Array<{
  focus: TourFocus;
  title: string;
  body: string;
}> = [
  {
    focus: "context",
    title: "Read the student context",
    body: "This section shows a student profile and the recent student-teacher conversation. Use it to understand what the student needs next.",
  },
  {
    focus: "interventions",
    title: "Compare both responses",
    body: "You will see two possible AI tutoring responses. Choose the one that would best help the student in the conversation you just read.",
  },
  {
    focus: "tie",
    title: "Use Tie when needed",
    body: "If both options seem equally helpful, or you cannot confidently choose one, select Mark Tie.",
  },
  {
    focus: "reasoning",
    title: "Add an optional note",
    body: "If you want to help improve the study, briefly explain why you chose your answer. Then click Submit & Next to move on.",
  },
];

const mockScenario: Scenario = {
  id: "mock_example",
  studentProfile: "First-Year Introductory Learner",
  chatContext:
    "student: I understand that a loop repeats code, but I do not know when it stops.\nteacher: What part of the loop tells the computer whether to keep going?\nstudent: I think it has something to do with the condition, but I am not sure how to check it.",
  interventionA:
    "hint: Focus on the condition first. Ask whether it is true before each repetition.",
  interventionB:
    "socratic_prompt: What would happen if the condition became false before the loop began?",
};

function getSessionId() {
  const existing = localStorage.getItem(SESSION_KEY);
  if (existing) return existing;

  const id = crypto.randomUUID ? crypto.randomUUID() : fallbackUuid();
  localStorage.setItem(SESSION_KEY, id);
  return id;
}

function fallbackUuid() {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0"));
  return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex
    .slice(6, 8)
    .join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`;
}

function getStoredEvaluatorRole(): EvaluatorRole | "" {
  const storedRole = localStorage.getItem(EVALUATOR_ROLE_KEY);
  return evaluatorRoles.includes(storedRole as EvaluatorRole)
    ? (storedRole as EvaluatorRole)
    : "";
}

function getStoredReminderPreference() {
  return localStorage.getItem(QUALITY_REMINDER_KEY) === "true";
}

function shuffleScenarios(source: Scenario[]) {
  const shuffled = [...source];
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const randomIndex = Math.floor(Math.random() * (index + 1));
    [shuffled[index], shuffled[randomIndex]] = [
      shuffled[randomIndex],
      shuffled[index],
    ];
  }
  return shuffled;
}

function Transcript({ value }: { value: string }) {
  const lines = value.split("\n").filter(Boolean);

  return (
    <div className="transcript" aria-label="Chat transcript">
      {lines.map((line, index) => {
        const [speaker, ...body] = line.split(":");
        return (
          <div className="turn" key={`${speaker}-${index}`}>
            <span className="speaker">{speaker}</span>
            <p>{body.join(":").trim()}</p>
          </div>
        );
      })}
    </div>
  );
}

function formatInterventionText(rawText: string) {
  return rawText.replace(/^[^:]+:\s+/, "");
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function calculateGuidePosition(
  focus: TourFocus,
  targetRect: DOMRect,
  popoverRect: DOMRect,
): CSSProperties {
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  const popoverWidth = popoverRect.width || Math.min(380, viewportWidth - 36);
  const popoverHeight = popoverRect.height || 210;
  const maxLeft = viewportWidth - popoverWidth - GUIDE_MARGIN;
  const maxTop = viewportHeight - popoverHeight - GUIDE_MARGIN;
  const centeredTop = targetRect.top + targetRect.height / 2 - popoverHeight / 2;
  const centeredLeft = targetRect.left + targetRect.width / 2 - popoverWidth / 2;

  if (viewportWidth <= 560) {
    if (focus === "interventions") {
      return {
        top: "auto",
        left: `${GUIDE_MARGIN}px`,
        right: `${GUIDE_MARGIN}px`,
        bottom: `${GUIDE_MARGIN}px`,
        width: "auto",
        transform: "none",
      };
    }

    const preferredTop =
      focus === "reasoning"
        ? targetRect.top - popoverHeight - GUIDE_GAP
        : targetRect.bottom + GUIDE_GAP;

    const fallbackTop =
      focus === "reasoning"
        ? targetRect.bottom + GUIDE_GAP
        : targetRect.top - popoverHeight - GUIDE_GAP;

    const top =
      preferredTop >= GUIDE_MARGIN && preferredTop <= maxTop
        ? preferredTop
        : fallbackTop;

    return {
      top: `${clamp(top, GUIDE_MARGIN, maxTop)}px`,
      left: `${GUIDE_MARGIN}px`,
      right: `${GUIDE_MARGIN}px`,
      bottom: "auto",
      width: "auto",
      transform: "none",
    };
  }

  if (focus === "context") {
    const rightSideLeft = targetRect.right + GUIDE_GAP;
    const left =
      rightSideLeft + popoverWidth <= viewportWidth - GUIDE_MARGIN
        ? rightSideLeft
        : targetRect.left - popoverWidth - GUIDE_GAP;

    return {
      top: `${clamp(centeredTop, GUIDE_MARGIN, maxTop)}px`,
      left: `${clamp(left, GUIDE_MARGIN, maxLeft)}px`,
      right: "auto",
      bottom: "auto",
      transform: "none",
    };
  }

  if (focus === "interventions") {
    const leftSideLeft = targetRect.left - popoverWidth - GUIDE_GAP;
    const left =
      leftSideLeft >= GUIDE_MARGIN
        ? leftSideLeft
        : targetRect.right + GUIDE_GAP;

    return {
      top: `${clamp(centeredTop, GUIDE_MARGIN, maxTop)}px`,
      left: `${clamp(left, GUIDE_MARGIN, maxLeft)}px`,
      right: "auto",
      bottom: "auto",
      transform: "none",
    };
  }

  const preferredTop = targetRect.top - popoverHeight - GUIDE_GAP;
  const fallbackTop = targetRect.bottom + GUIDE_GAP;
  const top =
    preferredTop >= GUIDE_MARGIN && preferredTop <= maxTop
      ? preferredTop
      : fallbackTop;

  return {
    top: `${clamp(top, GUIDE_MARGIN, maxTop)}px`,
    left: `${clamp(centeredLeft, GUIDE_MARGIN, maxLeft)}px`,
    right: "auto",
    bottom: "auto",
    transform: "none",
  };
}

type EvaluationFrameProps = {
  scenario: Scenario;
  selected: Selection | null;
  stageLabel: string;
  title: string;
  progressLabel: string;
  progress: number;
  isMock?: boolean;
  tourFocus?: TourFocus | null;
  onHelp?: () => void;
  onSelect: (choice: Selection) => void;
};

function EvaluationFrame({
  scenario,
  selected,
  stageLabel,
  title,
  progressLabel,
  progress,
  isMock = false,
  tourFocus = null,
  onHelp,
  onSelect,
}: EvaluationFrameProps) {
  return (
    <>
      <header className="topbar">
        <div>
          <p className="kicker">{stageLabel}</p>
          <h1>{title}</h1>
        </div>
        <div className={`topbar-actions ${onHelp ? "has-help" : ""}`}>
          {onHelp && (
            <button
              type="button"
              className="help-button"
              aria-label="Open guided tour"
              onClick={onHelp}
            >
              ?
            </button>
          )}
          <div className="progress-block" aria-label="Evaluation progress">
            <span>{progressLabel}</span>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
          </div>
        </div>
      </header>

      {isMock && (
        <aside className="mock-banner" role="status">
          MOCK EXAMPLE
        </aside>
      )}

      <section className="workspace">
        <article
          className={`context-panel field-profile ${
            tourFocus === "context" ? "guide-target" : ""
          }`}
        >
          <div className="panel-heading">
            <span>Student Profile</span>
            <strong>{scenario.studentProfile}</strong>
          </div>
          <div className="field-transcript">
            <Transcript value={scenario.chatContext} />
          </div>
        </article>

        <section
          className={`comparison field-comparison ${
            tourFocus === "interventions" ? "guide-target" : ""
          }`}
          aria-label="Intervention comparison"
        >
          <InterventionCard
            label="A"
            text={formatInterventionText(scenario.interventionA)}
            selected={selected === "A"}
            onSelect={() => onSelect("A")}
          />
          <InterventionCard
            label="B"
            text={formatInterventionText(scenario.interventionB)}
            selected={selected === "B"}
            onSelect={() => onSelect("B")}
          />
          <button
            type="button"
            className={`tie-choice-button ${selected === "Tie" ? "selected" : ""} ${
              tourFocus === "tie" ? "guide-target" : ""
            }`}
            onClick={() => onSelect("Tie")}
          >
            Mark Tie
          </button>
        </section>
      </section>
    </>
  );
}

function InterventionCard({
  label,
  text,
  selected,
  onSelect,
}: {
  label: "A" | "B";
  text: string;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <article className={`intervention ${selected ? "selected" : ""}`}>
      <div className="intervention-label">Intervention {label}</div>
      <p>{text}</p>
      <button
        type="button"
        className="choice-button"
        onClick={onSelect}
      >
        Select {label}
      </button>
    </article>
  );
}

function GuidedTourOverlay({
  stepIndex,
  finalLabel = "Finish Tour",
  style,
  onBack,
  onNext,
  onClose,
}: {
  stepIndex: number;
  finalLabel?: string;
  style?: CSSProperties;
  onBack: () => void;
  onNext: () => void;
  onClose: () => void;
}) {
  const step = tourSteps[stepIndex];
  const isLastStep = stepIndex === tourSteps.length - 1;

  return (
    <section
      className={`guide-popover guide-popover-${step.focus}`}
      style={style}
      role="dialog"
      aria-modal="true"
      aria-labelledby="guided-tour-title"
    >
      <button
        type="button"
        className="guide-close"
        aria-label="Close guided tour"
        onClick={onClose}
      >
        x
      </button>
      <p className="guide-step-count">Step {stepIndex + 1} of {tourSteps.length}</p>
      <h2 id="guided-tour-title">{step.title}</h2>
      <p>{step.body}</p>
      <div className="guide-actions">
        <button
          type="button"
          className="secondary-action"
          disabled={stepIndex === 0}
          onClick={onBack}
        >
          Back
        </button>
        <button type="button" className="primary-action compact" onClick={onNext}>
          {isLastStep ? finalLabel : "Next"}
        </button>
      </div>
    </section>
  );
}

export function App() {
  const [sessionId] = useState(getSessionId);
  const [shuffledScenarios] = useState(() => shuffleScenarios(scenarios));
  const [stage, setStage] = useState<Stage>("landing");
  const [currentIndex, setCurrentIndex] = useState(0);
  const [evaluatorRole, setEvaluatorRole] = useState<EvaluatorRole | "">(
    getStoredEvaluatorRole,
  );
  const [selected, setSelected] = useState<Selection | null>(null);
  const [reasoning, setReasoning] = useState("");
  const [feedbackText, setFeedbackText] = useState("");
  const [responseTimeMs, setResponseTimeMs] = useState<number | null>(null);
  const [warningLabel, setWarningLabel] = useState<WarningLabel | null>(null);
  const [pendingSelection, setPendingSelection] = useState<Selection | null>(null);
  const [showQualityReminder, setShowQualityReminder] = useState(false);
  const [showPedagogyDefinition, setShowPedagogyDefinition] = useState(false);
  const [tourStepIndex, setTourStepIndex] = useState(0);
  const [helpTourStepIndex, setHelpTourStepIndex] = useState<number | null>(null);
  const [guidePopoverStyle, setGuidePopoverStyle] = useState<CSSProperties>({});
  const [reminderCountdown, setReminderCountdown] = useState(
    QUALITY_REMINDER_SECONDS,
  );
  const [hideQualityReminder, setHideQualityReminder] = useState(
    getStoredReminderPreference,
  );
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isAdvancing, setIsAdvancing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scenarioStartedAtRef = useRef(Date.now());

  const currentScenario = shuffledScenarios[currentIndex];
  const progress = useMemo(
    () => Math.round(((currentIndex + 1) / shuffledScenarios.length) * 100),
    [currentIndex, shuffledScenarios.length],
  );

  const updateEvaluatorRole = (role: EvaluatorRole) => {
    setEvaluatorRole(role);
    localStorage.setItem(EVALUATOR_ROLE_KEY, role);
  };

  const activeGuideStepIndex =
    stage === "tour" ? tourStepIndex : helpTourStepIndex;
  const activeGuideFocus =
    activeGuideStepIndex === null
      ? null
      : tourSteps[activeGuideStepIndex].focus;

  const advanceTutorialTour = () => {
    if (tourStepIndex >= tourSteps.length - 1) {
      setTourStepIndex(0);
      setSelected(null);
      setStage("prestart");
      window.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }

    setTourStepIndex((index) => index + 1);
  };

  const goBackTutorialTour = () => {
    setTourStepIndex((index) => Math.max(0, index - 1));
  };

  const advanceHelpTour = () => {
    if (helpTourStepIndex === null) return;

    if (helpTourStepIndex >= tourSteps.length - 1) {
      setHelpTourStepIndex(null);
      return;
    }

    setHelpTourStepIndex((index) => (index === null ? 0 : index + 1));
  };

  const goBackHelpTour = () => {
    setHelpTourStepIndex((index) =>
      index === null ? null : Math.max(0, index - 1),
    );
  };

  useEffect(() => {
    if (!activeGuideFocus) return;

    window.setTimeout(() => {
      document
        .querySelector(".guide-target")
        ?.scrollIntoView({ block: "start", behavior: "smooth" });
    }, 0);
  }, [activeGuideFocus]);

  useEffect(() => {
    if (!activeGuideFocus) {
      setGuidePopoverStyle({});
      return;
    }

    let frameId = 0;

    const updateGuidePosition = () => {
      frameId = window.requestAnimationFrame(() => {
        const target = document.querySelector(".guide-target");
        const popover = document.querySelector(".guide-popover");
        if (!target || !popover) return;

        setGuidePopoverStyle(
          calculateGuidePosition(
            activeGuideFocus,
            target.getBoundingClientRect(),
            popover.getBoundingClientRect(),
          ),
        );
      });
    };

    updateGuidePosition();
    window.addEventListener("resize", updateGuidePosition);
    window.addEventListener("scroll", updateGuidePosition, true);

    return () => {
      window.cancelAnimationFrame(frameId);
      window.removeEventListener("resize", updateGuidePosition);
      window.removeEventListener("scroll", updateGuidePosition, true);
    };
  }, [activeGuideFocus]);

  useEffect(() => {
    if (stage !== "evaluation") return;
    scenarioStartedAtRef.current = Date.now();
    setResponseTimeMs(null);
    setWarningLabel(null);
    setPendingSelection(null);
    setShowQualityReminder(false);
    setReminderCountdown(QUALITY_REMINDER_SECONDS);
  }, [currentIndex, stage]);

  useEffect(() => {
    const shouldWarnBeforeLeaving =
      stage === "tour" ||
      stage === "prestart" ||
      stage === "evaluation" ||
      stage === "feedback";
    if (!shouldWarnBeforeLeaving) return;

    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue =
        "Your in-progress study responses may not be saved if you leave or refresh this page.";
    };

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, [stage]);

  useEffect(() => {
    if (!showQualityReminder || reminderCountdown <= 0) return;
    const timer = window.setTimeout(() => {
      setReminderCountdown((seconds) => seconds - 1);
    }, 1000);
    return () => window.clearTimeout(timer);
  }, [showQualityReminder, reminderCountdown]);

  const selectIntervention = (choice: Selection) => {
    if (stage === "evaluation") {
      const elapsedMs = Date.now() - scenarioStartedAtRef.current;
      const isFastResponse = elapsedMs < FAST_RESPONSE_THRESHOLD_MS;
      setResponseTimeMs(elapsedMs);
      setWarningLabel(isFastResponse ? "answered_under_3_seconds" : null);

      if (isFastResponse && !hideQualityReminder) {
        setPendingSelection(choice);
        setReminderCountdown(QUALITY_REMINDER_SECONDS);
        setShowQualityReminder(true);
        setError(null);
        return;
      }
    }

    setSelected(choice);
    setError(null);
    window.setTimeout(() => {
      document
        .querySelector(".reasoning-drawer")
        ?.scrollIntoView({ block: "end", behavior: "smooth" });
    }, 0);
  };

  const continueAfterQualityReminder = () => {
    if (!pendingSelection || reminderCountdown > 0) return;

    if (hideQualityReminder) {
      localStorage.setItem(QUALITY_REMINDER_KEY, "true");
    }

    setSelected(pendingSelection);
    setPendingSelection(null);
    setShowQualityReminder(false);
    setError(null);
  };

  const submitEvaluation = async (event: FormEvent) => {
    event.preventDefault();
    if (!currentScenario || !selected || isSubmitting) return;

    if (!isSupabaseConfigured || !supabase) {
      setError("Supabase is missing VITE_SUPABASE_ANON_KEY.");
      return;
    }

    setIsSubmitting(true);
    setError(null);

    const { error: insertError } = await supabase
      .from("human_evaluations")
      .insert({
        session_id: sessionId,
        scenario_id: currentScenario.id,
        selected_intervention: selected,
        evaluator_role: evaluatorRole,
        reasoning: reasoning.trim() || null,
        response_time_ms: responseTimeMs,
        warning_label: warningLabel,
      });

    if (insertError) {
      setError(insertError.message);
      setIsSubmitting(false);
      return;
    }

    setIsAdvancing(true);
    window.setTimeout(() => {
      if (currentIndex + 1 >= shuffledScenarios.length) {
        setStage("feedback");
      } else {
        setCurrentIndex((index) => index + 1);
      }
      setSelected(null);
      setReasoning("");
      setIsSubmitting(false);
      setIsAdvancing(false);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }, 180);
  };

  const submitFeedback = async (event: FormEvent) => {
    event.preventDefault();

    if (!isSupabaseConfigured || !supabase) {
      setError("Supabase is missing VITE_SUPABASE_ANON_KEY.");
      return;
    }

    setIsSubmitting(true);
    setError(null);

    const { error: insertError } = await supabase.from("human_feedback").insert({
      session_id: sessionId,
      feedback_text: feedbackText.trim() || null,
    });

    if (insertError) {
      setError(insertError.message);
      setIsSubmitting(false);
      return;
    }

    setStage("finished");
    setIsSubmitting(false);
  };

  if (stage === "landing") {
    return (
      <main className="intro-screen">
        <section className="intro-copy">
          <p className="kicker">Human Preference Study</p>
          <h1>Welcome.</h1>
          <p>
            This study evaluates AI tutoring interventions for educational
            dialogue. You will be presented with 30 brief scenarios showing a
            student's profile and chat history. Your task is to select which AI
            intervention is{" "}
            <span className="definition-wrapper">
              <button
                type="button"
                className="definition-trigger"
                aria-expanded={showPedagogyDefinition}
                aria-describedby="pedagogy-definition"
                onClick={() =>
                  setShowPedagogyDefinition((isVisible) => !isVisible)
                }
              >
                pedagogically
              </button>
              {showPedagogyDefinition && (
                <span
                  id="pedagogy-definition"
                  className="definition-popover"
                  role="tooltip"
                >
                  Related to teaching quality: how well an intervention helps a
                  student understand, reason, and continue learning.
                </span>
              )}
            </span>{" "}
            superior.
          </p>
          <div className="disclaimer-block">
            <p>
              All responses are recorded completely anonymously to ensure
              unbiased, honest feedback.
            </p>
            <p>
              This is a double-blind study; the AI models generating the
              responses are hidden and randomized.
            </p>
          </div>
          <fieldset className="role-selector">
            <legend>Evaluator Role</legend>
            <div className="role-options">
              {evaluatorRoles.map((role) => (
                <label key={role} className="role-option">
                  <input
                    type="radio"
                    name="evaluator-role"
                    value={role}
                    checked={evaluatorRole === role}
                    onChange={() => updateEvaluatorRole(role)}
                    required
                  />
                  <span>{role}</span>
                </label>
              ))}
            </div>
          </fieldset>
          <button
            type="button"
            className="primary-action"
            disabled={!evaluatorRole}
            onClick={() => setStage("tour")}
          >
            Start Tutorial
          </button>
        </section>
      </main>
    );
  }

  if (stage === "tour") {
    return (
      <main className={`app-shell guide-mode guide-focus-${activeGuideFocus}`}>
        <EvaluationFrame
          scenario={mockScenario}
          selected={selected}
          stageLabel="Tutorial"
          title="Mock Example"
          progressLabel="Practice only"
          progress={0}
          isMock
          tourFocus={activeGuideFocus}
          onSelect={selectIntervention}
        />
        <form className={`reasoning-drawer open ${activeGuideFocus === "reasoning" ? "guide-target" : ""}`}>
          <div className="drawer-grid">
            <label htmlFor="mock-reasoning">
              Briefly, why did you choose this? <span>(Optional)</span>
            </label>
            <textarea
              id="mock-reasoning"
              value=""
              readOnly
              placeholder="Short rationale"
            />
            <div className="drawer-actions">
              <button type="button" className="submit-button">
                Submit & Next
              </button>
            </div>
          </div>
        </form>
        <GuidedTourOverlay
          stepIndex={tourStepIndex}
          finalLabel="I understand"
          style={guidePopoverStyle}
          onBack={goBackTutorialTour}
          onNext={advanceTutorialTour}
          onClose={() => {
            setTourStepIndex(0);
            setSelected(null);
            setStage("prestart");
            window.scrollTo({ top: 0, behavior: "smooth" });
          }}
        />
      </main>
    );
  }

  if (stage === "prestart") {
    return (
      <main className="intro-screen prestart-screen">
        <section className="intro-copy">
          <p className="kicker">Before You Begin</p>
          <h1>Ready to start.</h1>
          <div className="disclaimer-block">
            <p>
              We recommend completing the survey in a quiet, distraction-free
              setting. Distractions can affect the quality of the data.
            </p>
            <p>
              The survey usually takes 5-10 minutes. Please complete it in one
              sitting; closing or refreshing the tab may leave answers unsaved.
            </p>
            <p>
              Thank you for volunteering your time to help improve the future
              of education.
            </p>
          </div>
          <button
            type="button"
            className="primary-action"
            onClick={() => {
              setSelected(null);
              setError(null);
              scenarioStartedAtRef.current = Date.now();
              setStage("evaluation");
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
          >
            Begin Survey
          </button>
        </section>
      </main>
    );
  }

  if (stage === "feedback") {
    return (
      <main className="feedback-screen">
        <form className="feedback-form" onSubmit={submitFeedback}>
          <p className="kicker">Debrief</p>
          <h1>Evaluation Complete.</h1>
          <p>Thank you for your contribution to this research.</p>
          <label htmlFor="feedback">
            Do you have any general feedback on the study, the interventions,
            or the AI's pedagogical choices?
          </label>
          <textarea
            id="feedback"
            value={feedbackText}
            maxLength={8000}
            onChange={(event) => setFeedbackText(event.target.value)}
            placeholder="Optional feedback"
          />
          <button type="submit" className="primary-action" disabled={isSubmitting}>
            {isSubmitting ? "Submitting" : "Submit Feedback & Finish"}
          </button>
          {error && <p className="error-message standalone">{error}</p>}
        </form>
      </main>
    );
  }

  if (stage === "finished") {
    return (
      <main className="complete-screen">
        <p className="kicker">Human Evaluation</p>
        <h1>Finished.</h1>
        <p>Your responses and feedback have been recorded.</p>
      </main>
    );
  }

  return (
    <main
      className={`app-shell ${isAdvancing ? "is-advancing" : ""} ${
        helpTourStepIndex !== null ? `guide-mode guide-focus-${activeGuideFocus}` : ""
      }`}
    >
      <EvaluationFrame
        scenario={currentScenario}
        selected={selected}
        stageLabel="Blind A/B Tutoring Evaluation"
        title={`Scenario ${currentIndex + 1} of ${shuffledScenarios.length}`}
        progressLabel={`${progress}% complete`}
        progress={progress}
        tourFocus={activeGuideFocus}
        onHelp={() => setHelpTourStepIndex(0)}
        onSelect={selectIntervention}
      />

      {helpTourStepIndex !== null && (
        <GuidedTourOverlay
          stepIndex={helpTourStepIndex}
          style={guidePopoverStyle}
          onBack={goBackHelpTour}
          onNext={advanceHelpTour}
          onClose={() => setHelpTourStepIndex(null)}
        />
      )}

      {showQualityReminder && (
        <div className="modal-backdrop" role="presentation">
          <section
            className="quality-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="quality-reminder-title"
          >
            <p className="kicker">Reading Check</p>
            <h2 id="quality-reminder-title">Please take your time.</h2>
            <p>
              Careful judgments help improve future tutoring tools for{" "}
              people in roles like yours. Please spend at least a few seconds
              reading both choices before continuing.
            </p>
            <label className="modal-checkbox">
              <input
                type="checkbox"
                checked={hideQualityReminder}
                onChange={(event) => {
                  const shouldHide = event.target.checked;
                  setHideQualityReminder(shouldHide);
                  if (!shouldHide) {
                    localStorage.removeItem(QUALITY_REMINDER_KEY);
                  }
                }}
              />
              <span>Do not remind me again</span>
            </label>
            <button
              type="button"
              className="primary-action compact"
              disabled={reminderCountdown > 0}
              onClick={continueAfterQualityReminder}
            >
              {reminderCountdown > 0
                ? `Continue in ${reminderCountdown}`
                : "Close and Continue"}
            </button>
          </section>
        </div>
      )}

      {!isSupabaseConfigured && (
        <aside className="system-notice" role="status">
          Add <code>VITE_SUPABASE_ANON_KEY</code> to your local env before
          collecting responses.
        </aside>
      )}

      <form
        className={`reasoning-drawer ${
          selected || activeGuideFocus === "reasoning" ? "open" : ""
        } ${activeGuideFocus === "reasoning" ? "guide-target" : ""}`}
        onSubmit={submitEvaluation}
      >
        <div className="drawer-grid">
          <label htmlFor="reasoning">
            Briefly, why did you choose this? <span>(Optional)</span>
          </label>
          <textarea
            id="reasoning"
            value={reasoning}
            maxLength={4000}
            onChange={(event) => setReasoning(event.target.value)}
            placeholder="Short rationale"
          />
          <div className="drawer-actions">
            <button
              type="submit"
              className="submit-button"
              disabled={!selected || isSubmitting}
            >
              {isSubmitting ? "Submitting" : `Submit & Next`}
            </button>
          </div>
        </div>
        {error && <p className="error-message">{error}</p>}
      </form>
    </main>
  );
}
