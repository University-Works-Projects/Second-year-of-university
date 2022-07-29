<script>
export default {
  data() {
    return {
      text: ''
    }
  },
  methods: {
    onInput(e) {
      this.text = e.target.value
    }
  }
}
</script>

<template>
  <input v-model='text' placeholder="Type here">
  <input :value="text" @input="onInput" placeholder="Type here"> <!-- Shortcut -->
  <p>{{ text }}</p>
</template>