/**
 * fuzzyart.js
 * Motor de Fuzzy ART para autenticación biométrica por dinámica de tecleo.
 * Traducción directa (numérica) de la clase Python del notebook Modelo__FuzzyArt.ipynb.
 *
 * Diseñado para ejecutarse en el navegador: sin dependencias externas.
 * Los resultados numéricos coinciden con los de scikit-learn/NumPy porque
 * las operaciones son elementales (min, sum, división).
 */

// =============================================================================
// MinMaxScaler equivalente al de scikit-learn, con clipping en transform().
// Solo maneja el caso feature_range=(0,1), que es el que usa el notebook.
// =============================================================================
class MinMaxScaler {
  constructor() {
    this.min = null; // vector de mínimos por feature
    this.max = null; // vector de máximos por feature
  }

  fit(X) {
    if (X.length === 0) throw new Error("MinMaxScaler.fit: dataset vacío");
    const n = X[0].length;
    this.min = new Array(n).fill(Infinity);
    this.max = new Array(n).fill(-Infinity);
    for (const row of X) {
      for (let j = 0; j < n; j++) {
        if (row[j] < this.min[j]) this.min[j] = row[j];
        if (row[j] > this.max[j]) this.max[j] = row[j];
      }
    }
    return this;
  }

  transform(X) {
    return X.map(row => row.map((v, j) => {
      const range = this.max[j] - this.min[j];
      // Si una feature es constante (range=0) el resultado se define como 0
      // (igual que sklearn en la práctica), y luego se clipea igual.
      const scaled = range === 0 ? 0 : (v - this.min[j]) / range;
      // Clip [0,1] — necesario porque valores en test pueden salir del rango de fit.
      return Math.max(0, Math.min(1, scaled));
    }));
  }

  fitTransform(X) { return this.fit(X).transform(X); }

  toJSON() {
    return { min: this.min, max: this.max };
  }

  static fromJSON(obj) {
    const s = new MinMaxScaler();
    s.min = obj.min;
    s.max = obj.max;
    return s;
  }
}

// =============================================================================
// FuzzyART — traducción directa del notebook.
// - rho:   vigilancia (0..1), qué tan estricto es el match
// - alpha: estabilidad numérica en la fórmula de elección
// - beta:  velocidad de aprendizaje (1.0 = fast learning)
// =============================================================================
class FuzzyART {
  constructor({ rho = 0.897, alpha = 0.0001, beta = 0.5 } = {}) {
    this.rho = rho;
    this.alpha = alpha;
    this.beta = beta;
    this.w = [];            // prototipos: cada uno es un vector de 2*n_features
    this.num_categories = 0;
  }

  _complementCoding(x) {
    // [x, 1-x] concatenado
    const out = new Array(x.length * 2);
    for (let i = 0; i < x.length; i++) {
      out[i] = x[i];
      out[i + x.length] = 1 - x[i];
    }
    return out;
  }

  _minVec(a, b) {
    const out = new Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] < b[i] ? a[i] : b[i];
    return out;
  }

  _sum(a) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i];
    return s;
  }

  /**
   * Entrena con un batch de muestras (o incrementalmente si ya hay categorías).
   * X debe estar YA normalizado a [0,1] (usa el MinMaxScaler antes).
   * Este método es idempotente en el sentido de que llamarlo múltiples veces
   * con nuevas muestras produce el mismo resultado que entrenarlas todas juntas
   * — es la propiedad de aprendizaje incremental de Fuzzy ART.
   */
  train(X) {
    if (X.length === 0) return;
    const Xcomp = X.map(x => this._complementCoding(x));
    let startIdx = 0;

    if (this.num_categories === 0) {
      // Primera muestra siempre inicializa una categoría (no tiene con qué compararse).
      this.w.push(Xcomp[0].slice());
      this.num_categories += 1;
      startIdx = 1;
    }

    for (let i = startIdx; i < Xcomp.length; i++) {
      const I = Xcomp[i];
      const normI = this._sum(I);
      let matched = false;

      // Ranking de candidatos por función de elección T_k
      const candidates = new Array(this.num_categories);
      for (let k = 0; k < this.num_categories; k++) {
        const wk = this.w[k];
        const intersection = this._minVec(I, wk);
        const normInt = this._sum(intersection);
        const normW = this._sum(wk);
        candidates[k] = { t: normInt / (this.alpha + normW), k, intersection, normInt };
      }
      candidates.sort((a, b) => b.t - a.t);

      // Prueba de resonancia contra cada candidato en orden decreciente de T
      for (const c of candidates) {
        if (normI === 0) break; // seguridad ante vectores todo-cero
        if ((c.normInt / normI) >= this.rho) {
          // Aprende: refina el prototipo
          const wk = this.w[c.k];
          const inter = c.intersection;
          for (let j = 0; j < wk.length; j++) {
            wk[j] = this.beta * inter[j] + (1 - this.beta) * wk[j];
          }
          matched = true;
          break;
        }
      }

      if (!matched) {
        this.w.push(I.slice());
        this.num_categories += 1;
      }
    }
  }

  /**
   * Predice 1 (autenticado) o 0 (rechazado) por cada muestra.
   * X debe estar YA normalizado a [0,1].
   */
  predict(X) {
    return X.map(x => {
      const I = this._complementCoding(x);
      const normI = this._sum(I);
      if (normI === 0) return 0;
      for (let k = 0; k < this.num_categories; k++) {
        const wk = this.w[k];
        const inter = this._minVec(I, wk);
        const normInt = this._sum(inter);
        if ((normInt / normI) >= this.rho) return 1;
      }
      return 0;
    });
  }

  toJSON() {
    return {
      rho: this.rho,
      alpha: this.alpha,
      beta: this.beta,
      num_categories: this.num_categories,
      w: this.w,
    };
  }

  static fromJSON(obj) {
    const m = new FuzzyART({ rho: obj.rho, alpha: obj.alpha, beta: obj.beta });
    m.w = obj.w.map(v => v.slice());
    m.num_categories = obj.num_categories;
    return m;
  }
}

// =============================================================================
// Política de autenticación por votación deslizante.
// Copia exacta de voting_authentication() del notebook.
// Devuelve el porcentaje de ventanas que superaron el umbral (útil para evaluar
// TAR/FAR globales) — pero en el flujo web usamos la versión "una decisión"
// más abajo, para dar veredicto tras N intentos consecutivos del usuario.
// =============================================================================
function votingAuthentication(preds, windowSize, threshold) {
  const decisions = [];
  for (let i = 0; i <= preds.length - windowSize; i++) {
    let sum = 0;
    for (let j = i; j < i + windowSize; j++) sum += preds[j];
    decisions.push(sum >= threshold ? 1 : 0);
  }
  if (decisions.length === 0) return 0;
  return decisions.reduce((a, b) => a + b, 0) / decisions.length * 100;
}

/**
 * Veredicto sobre una sola sesión de N intentos del usuario final.
 * Ej: N=7 intentos, umbral=4 => acepta si acierta ≥4 de 7.
 */
function singleSessionVerdict(preds, threshold) {
  const hits = preds.reduce((a, b) => a + b, 0);
  return { hits, total: preds.length, accepted: hits >= threshold };
}

// =============================================================================
// Envoltorio: modelo por usuario, incluye scaler + fuzzyart + config de votación.
// Este es el objeto que se serializa a JSON y se guarda en Drive.
// =============================================================================
class UserModel {
  constructor(subject, config = {}) {
    this.subject = subject;
    this.version = 1;
    this.created_at = new Date().toISOString();
    this.updated_at = this.created_at;
    this.n_samples_trained = 0;
    this.hyperparameters = {
      rho:   config.rho   ?? 0.897,
      alpha: config.alpha ?? 0.0001,
      beta:  config.beta  ?? 0.5,
    };
    this.voting = {
      window_size: config.window_size ?? 7,
      threshold:   config.threshold   ?? 4,
    };
    this.scaler = new MinMaxScaler();
    this.art = new FuzzyART(this.hyperparameters);
    this._scalerFitted = false;
  }

  /**
   * Ajusta scaler y entrena Fuzzy ART con el batch inicial de muestras.
   * Debe llamarse UNA sola vez, con al menos MIN_SAMPLES muestras.
   */
  initialTrain(samples) {
    this.scaler.fit(samples);
    this._scalerFitted = true;
    const normalized = this.scaler.transform(samples);
    this.art.train(normalized);
    this.n_samples_trained = samples.length;
    this.updated_at = new Date().toISOString();
  }

  /**
   * Actualiza incrementalmente con una única muestra nueva.
   * NOTA sobre scaler: no se recalcula (fijar min/max estables es requisito
   * para que Fuzzy ART incremental sea coherente). Valores fuera de rango
   * se clipean a [0,1] dentro de transform().
   */
  incrementalUpdate(sample) {
    if (!this._scalerFitted) {
      throw new Error("UserModel: llama a initialTrain() antes de incrementalUpdate()");
    }
    const normalized = this.scaler.transform([sample]);
    this.art.train(normalized);
    this.n_samples_trained += 1;
    this.updated_at = new Date().toISOString();
  }

  /**
   * Corre la predicción sobre N intentos ya recogidos y devuelve veredicto.
   */
  authenticate(samples) {
    const normalized = this.scaler.transform(samples);
    const preds = this.art.predict(normalized);
    return singleSessionVerdict(preds, this.voting.threshold);
  }

  toJSON() {
    return {
      subject: this.subject,
      version: this.version,
      created_at: this.created_at,
      updated_at: this.updated_at,
      n_samples_trained: this.n_samples_trained,
      hyperparameters: this.hyperparameters,
      voting: this.voting,
      scaler: this.scaler.toJSON(),
      art: this.art.toJSON(),
    };
  }

  static fromJSON(obj) {
    const m = new UserModel(obj.subject, {
      ...obj.hyperparameters,
      window_size: obj.voting.window_size,
      threshold:   obj.voting.threshold,
    });
    m.version = obj.version;
    m.created_at = obj.created_at;
    m.updated_at = obj.updated_at;
    m.n_samples_trained = obj.n_samples_trained;
    m.scaler = MinMaxScaler.fromJSON(obj.scaler);
    m.art = FuzzyART.fromJSON(obj.art);
    m._scalerFitted = true;
    return m;
  }
}

// Exponer al scope global del navegador
window.MinMaxScaler = MinMaxScaler;
window.FuzzyART = FuzzyART;
window.UserModel = UserModel;
window.votingAuthentication = votingAuthentication;
window.singleSessionVerdict = singleSessionVerdict;
